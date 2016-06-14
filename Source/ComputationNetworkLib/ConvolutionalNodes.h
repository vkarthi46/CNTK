//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "ComputationNode.h"
#include "ConvolutionEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// ConvolutionNodeBase
// -----------------------------------------------------------------------

// ConvolutionNodeBase is a base class for ND-convolution(ConvolutionNode) and ND-pooling(PoolingNode).
// 
// 2D convolutions (incl. pooling) support two different storage formats:
//
// * legacy ("HWC") mode: Channels are tuples of scalars
//
//    This follows "high performance convolutional neural networks for document processing" by Kumar Chellapilla, Sidde Puri, and Patrice Simard.
//    Each sample is stored as a column-major matrix (height, width) of float[numChannels] (r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11).
//
//     - input :  [C  x W  x H      x T]  or  ARRAY[1..T] OF                ARRAY[1..H]  OF ARRAY[1..W]  OF ARRAY[1..C]
//     - output : [C' x W' x H'     x T]  or  ARRAY[1..T] OF                ARRAY[1..H'] OF ARRAY[1..W'] OF ARRAY[1..C']
//     - filter : [C' x W" x H" x C    ]  or                 ARRAY[1..C] OF ARRAY[1..H"] OF ARRAY[1..W"] OF ARRAY[1..C']
//
// * cudnn ("CHW") mode (works both GPU and CPU): Channels are planes
//
//     - input :   [W  x H  x C       x T]   or  ARRAY[1..T] OF                 ARRAY[1..C]  OF ARRAY[1..H]  OF ARRAY[1..W]
//     - output :  [W' x H' x      C' x T]   or  ARRAY[1..T] OF ARRAY[1..C'] OF                 ARRAY[1..H'] OF ARRAY[1..W']
//     - filter :  [W" x H" x C  x C'    ]   or                 ARRAY[1..C'] OF ARRAY[1..C]  OF ARRAY[1..H]  OF ARRAY[1..W]
//
// where:
//  - using ' for output and " for filter
//  - T = samples (NVidia calls this N)
//  - W, H = width, height (W', H' for output, W", H" for kernel)
//  - C = input channels
//     - 3 for color images, 1 for B&W images
//     - for hidden layer: dimension of activation vector for each pixel
//  - C' = output channels = dimension of activation vector for each pixel (also called N by NVidia, inconsistently)
//
// For ND-convolution/pooling only second format ('cudnn') is supported.
// 
template <class ElemType>
class ConvolutionNodeBase : public ComputationNode<ElemType>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;

public:
    ConvolutionNodeBase(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name), m_poolKind(PoolKind::None), m_maxTempMemSizeInSamples(0)
    {
    }
    ConvolutionNodeBase(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& kernelShape, const TensorShape& mapCount, const TensorShape& strideShape,
                    const std::vector<bool>& sharing, const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                    PoolKind poolKind, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples)
                    : Base(deviceId, name), m_kernelShape(kernelShape), m_mapCount(mapCount), m_stride(strideShape), m_sharing(sharing),
                    m_autoPad(autoPadding), m_lowerPad(lowerPad), m_upperPad(upperPad), m_poolKind(poolKind),
                    m_imageLayout(imageLayout), m_maxTempMemSizeInSamples(maxTempMemSizeInSamples)
    {
    }

public:
    void Save(File& fstream) const override
    {
        Base::Save(fstream);

        m_kernelShape.Save(fstream);
        m_mapCount.Save(fstream);
        m_stride.Save(fstream);
        fstream << m_sharing;
        fstream << m_autoPad;
        m_lowerPad.Save(fstream);
        m_upperPad.Save(fstream);
        fstream << (int32_t)m_poolKind;
        fstream << (int32_t)m_imageLayout;
        fstream << m_maxTempMemSizeInSamples;
    }

    void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);

        // Let ConvolutionNode handle older models.
        if (modelVersion >= CNTK_MODEL_VERSION_5)
        {
            m_kernelShape.Load(fstream);
            m_mapCount.Load(fstream);
            m_stride.Load(fstream);
            fstream >> m_sharing;
            fstream >> m_autoPad;
            m_lowerPad.Load(fstream);
            m_upperPad.Load(fstream);
            int32_t k;
            fstream >> k;
            m_poolKind = (PoolKind)k;
            int32_t layout;
            fstream >> layout;
            m_imageLayout = (ImageLayoutKind)layout;
            fstream >> m_maxTempMemSizeInSamples;
    }
    }

    void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<ConvolutionNodeBase<ElemType>>(nodeP);
            node->m_kernelShape = m_kernelShape;
            node->m_mapCount = m_mapCount;
            node->m_stride = m_stride;
            node->m_sharing = m_sharing;
            node->m_autoPad = m_autoPad;
            node->m_lowerPad = m_lowerPad;
            node->m_upperPad = m_upperPad;
            node->m_poolKind = m_poolKind;
            node->m_imageLayout = m_imageLayout;
            node->m_maxTempMemSizeInSamples = m_maxTempMemSizeInSamples;
        }
    }

    void BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        auto sliceOutputGrad = GradientFor(fr);

        if (m_poolKind == PoolKind::None)
        {
        if (inputIndex == 0) // derivative with respect to the weight matrix
        {
            auto& grad = Input(0)->GradientAsMatrix();
                auto sliceInput1Value = Input(1)->ValueFor(fr);
                m_convEng->BackwardKernel(sliceOutputGrad, sliceInput1Value, grad, fr.IsAllFrames(), *m_tempMatrix);
        }
        else if (inputIndex == 1) // derivative with respect to the input feature
        {
            auto& input0 = Input(0)->ValueAsMatrix();
            auto sliceInput1Grad = Input(1)->GradientFor(fr);
                m_convEng->BackwardData(sliceOutputGrad, input0, sliceInput1Grad, *m_tempMatrix);
        }
    }
        else
        {
            Matrix<ElemType> sliceInput0Grad = Input(0)->GradientFor(fr);

            Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
            Matrix<ElemType> sliceOutputValue = ValueFor(fr);

            m_convEng->BackwardPooling(sliceOutputValue, sliceOutputGrad, sliceInput0Value, sliceInput0Grad);
        }
    }

    bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The ConvolutionNode requires output values only for max pooling.
        return m_poolKind == PoolKind::Max;
    }

    void ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        if (m_poolKind == PoolKind::None)
        {
        const Matrix<ElemType>& input0 = Input(0)->ValueAsMatrix();
        Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);
            m_convEng->Forward(sliceInput1Value, input0, sliceOutputValue, *m_tempMatrix);
        }
        else
        {
            const Matrix<ElemType>& input0 = Input(0)->ValueFor(fr);
            m_convEng->ForwardPooling(input0, sliceOutputValue);
        }
    }

    void DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const override
    {
        Base::DumpNodeInfo(printValues, printMetadata, fstream);

        if (m_convEng != nullptr)
            fstream << "Geometry: " << string(*m_convEng->Geometry()) << "\n";
        fstream << "PoolKind: " << (int)m_poolKind << "\n";
    }

protected:
    TensorShape m_kernelShape;
    TensorShape m_mapCount;
    TensorShape m_stride;
    std::vector<bool> m_sharing;
    std::vector<bool> m_autoPad;
    TensorShape m_lowerPad;
    TensorShape m_upperPad;
    PoolKind m_poolKind;
    ImageLayoutKind m_imageLayout;

	size_t m_outH;
	size_t m_outW;

    size_t m_maxTempMemSizeInSamples;
    shared_ptr<Matrix<ElemType>> m_tempMatrix;

    std::unique_ptr<ConvolutionEngine<ElemType>> m_convEng;
};

#define UsingConvolutionNodeBaseMembers     \
    UsingComputationNodeMembersBoilerplate; \
protected:                                  \
    using Base::m_kernelShape;              \
    using Base::m_mapCount;                 \
    using Base::m_stride;                   \
    using Base::m_sharing;                  \
    using Base::m_autoPad;                  \
    using Base::m_lowerPad;                 \
    using Base::m_upperPad;                 \
    using Base::m_poolKind;                 \
    using Base::m_imageLayout;              \
    using Base::m_maxTempMemSizeInSamples;  \
    using Base::m_tempMatrix;               \
    using Base::m_convEng;                  \
public:

// -----------------------------------------------------------------------
// ConvolutionNode (convolutionWeights, inputFeature)
// -----------------------------------------------------------------------

template <class ElemType>
class ConvolutionNode : public ConvolutionNodeBase<ElemType>, public NumInputs<2>
{
    typedef ConvolutionNodeBase<ElemType> Base;
    UsingConvolutionNodeBaseMembers;
    static const std::wstring TypeName()
    {
        return L"Convolution";
    }

public:
    ConvolutionNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    ConvolutionNode(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& kernelShape, const TensorShape& mapCount, const TensorShape& strideShape,
                    const std::vector<bool>& sharing, const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                    ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples)
                    : Base(deviceId, name, kernelShape, mapCount, strideShape, sharing, autoPadding, lowerPad, upperPad, PoolKind::None, imageLayout, maxTempMemSizeInSamples),
                    m_convolution2D(false)
    {
    }
    ConvolutionNode(DEVICEID_TYPE deviceId, const wstring& name, const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels,
                    const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayout,
                    bool zeroPadding, size_t maxTempMemSizeInSamples)
                    : ConvolutionNode(deviceId, name, TensorShape(kernelWidth, kernelHeight, 1), TensorShape(1, 1, outputChannels),
                                      TensorShape(horizontalSubsample, verticalSubsample, 1), vector<bool>{true}, 
                                      vector<bool>{zeroPadding}, TensorShape(0), TensorShape(0),
                                      imageLayout, maxTempMemSizeInSamples)
    {
        m_convolution2D = true;
    }
    ConvolutionNode(const ScriptableObjects::IConfigRecordPtr configp)
        : ConvolutionNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"kernelShape"), configp->Get(L"mapCount"), configp->Get(L"strideShape"),
                          configp->Get(L"dimSharing"), configp->Get(L"dimPadding"), configp->Get(L"dimPadLower"), configp->Get(L"dimPadUpper"),
                          ImageLayoutKindFrom(configp->Get(L"imageLayout")), configp->Get(L"maxTempMemSizeInSamples"))
    {
        AttachInputsFromConfig(configp, GetExpectedNumInputs());
    }

public:
    void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_convolution2D;
    }

    void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);

        // Back compat: load pre-ND convolution models.
        if (modelVersion < CNTK_MODEL_VERSION_5)
        {
            size_t kW, kH, sW, sH;
            fstream >> kW;
            fstream >> kH;
            fstream >> sW;
            fstream >> sH;
            uint32_t imageLayout, mapCount;
            fstream >> mapCount;
            fstream >> imageLayout;
            m_imageLayout = (ImageLayoutKind)imageLayout;
            bool pad;
            fstream >> pad;
            fstream >> m_maxTempMemSizeInSamples;
            m_poolKind = PoolKind::None;
            m_convolution2D = true;

            m_kernelShape = TensorShape(kW, kH, 1);
            m_mapCount = TensorShape(mapCount);
            m_stride = TensorShape(sW, sH, 1);
            m_sharing = vector<bool>{true};
            m_autoPad = vector<bool>{pad};
            m_lowerPad = TensorShape(0);
            m_upperPad = TensorShape(0);
        }
        else
        {
            fstream >> m_convolution2D;
        }
    }

    void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<ConvolutionNode<ElemType>>(nodeP);
            node->m_convolution2D = m_convolution2D;
        }
    }

    void Validate(bool isFinalValidationPass) override
    {

        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        size_t inputIdx = GetExpectedNumInputs() - 1;
        TensorShape inputShape;
        if (m_convolution2D)
        {
            // Need to update some tensors with correct input dims.
            auto inDims = ImageDimensions(GetInputSampleLayout(inputIdx), m_imageLayout);
            // inputShape is used in ConvolveGeometry which supports only CHW layout.
            inputShape = inDims.AsTensorShape(ImageLayoutKind::CHW);
            size_t kW = m_kernelShape[0];
            size_t kH = m_kernelShape[1];
            size_t sW = m_stride[0];
            size_t sH = m_stride[1];
            m_kernelShape = TensorShape(kW, kH, inDims.m_numChannels);
            m_stride = TensorShape(sW, sH, inDims.m_numChannels);

            size_t mapCount = m_mapCount.GetNumElements();
            size_t weightCols = kW * kH * inDims.m_numChannels;

            // check/infer input [0] (weights)
            // BUGBUG: For now, we treat the weights as a 2D matrix. They should be a tensor proper.
            Input(0)->ValidateInferInputDimsFrom(TensorShape(mapCount, weightCols));

            if (isFinalValidationPass && (Input(0)->GetAsMatrixNumCols() != weightCols || Input(0)->GetAsMatrixNumRows() != mapCount))
            {
                LogicError("Convolution weight matrix %ls should have dimension [%d, %d] which is [outputChannels, kernelWidth * kernelHeight * inputChannels]", 
                           Input(0)->NodeName().c_str(), (int)mapCount, (int)weightCols);
            }

            auto outDims = ConvolveGeometry::ComputeOutputShape(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                                 m_sharing, m_autoPad, m_lowerPad, m_upperPad);
            // ConvolveGeometry always uses CHW.
            SetDims(ImageDimensions(outDims, ImageLayoutKind::CHW).AsTensorShape(m_imageLayout), HasMBLayout());
        }
        else
        {
            if (m_imageLayout != ImageLayoutKind::CHW)
            {
                InvalidArgument(
                    "%ls %ls supports only cuDNN (CHW) data layout. "
                    "Please specify imageLayout=\"cudnn\" in %ls node in your script "
                    "and make sure input data layout is CHW", NodeName().c_str(), OperationName().c_str(), NodeName().c_str());
            }
            inputShape = GetInputSampleLayout(inputIdx);
            auto outDims = ConvolveGeometry::ComputeOutputShape(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                                m_sharing, m_autoPad, m_lowerPad, m_upperPad);
            SetDims(outDims, HasMBLayout());
        }

        if (isFinalValidationPass)
        {
            if (m_convEng == nullptr)
            {
                auto geometry = std::make_shared<ConvolveGeometry>(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                                   m_sharing, m_autoPad, m_lowerPad, m_upperPad);
                m_convEng = ConvolutionEngine<ElemType>::Create(geometry, m_deviceId, m_imageLayout,
                                                                m_maxTempMemSizeInSamples, m_poolKind);
            }

            if (Input(0)->GetAsMatrixNumCols() != m_kernelShape.GetNumElements() ||
                Input(0)->GetAsMatrixNumRows() != m_convEng->Geometry()->KernelCount())
            {
                LogicError("Convolution weight matrix %ls should have dimension [%d, %d] which is [kernelCount, kernelWidth * kernelHeight * inputChannels]",
                           Input(0)->NodeName().c_str(), (int)m_convEng->Geometry()->KernelCount(), (int)m_kernelShape.GetNumElements());
            }
        }
    }

    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_tempMatrix, matrixPool);
    }

    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_tempMatrix, matrixPool);
    }

    void SetmMaxTempMemSizeInSamples(const size_t maxTempMemSizeInSamples)
    {
        m_maxTempMemSizeInSamples = maxTempMemSizeInSamples;
        if (m_convEng != nullptr)
            m_convEng->SetmMaxTempMemSizeInSamples(maxTempMemSizeInSamples);
    }

protected:
    bool m_convolution2D;
};


// -----------------------------------------------------------------------
// ROIPoolingNode (inputROIs, inputFeatures)
// -----------------------------------------------------------------------

template <class ElemType>
class ROIPoolingNode : public ComputationNode<ElemType>, public NumInputs<2>
{
	typedef ComputationNode<ElemType> Base;
	UsingComputationNodeMembersBoilerplate;

  static const std::wstring TypeName() 
  {
    return L"ROIPooling";
  }
public:

	ROIPoolingNode(DEVICEID_TYPE deviceId, const wstring& name)
		: Base(deviceId, name)
	{
	}
	ROIPoolingNode(DEVICEID_TYPE deviceId, const wstring& name, const size_t H, const size_t W, ImageLayoutKind imageLayoutKind)
		: Base(deviceId, name), m_outH(H), m_outW(W), m_imageLayout(imageLayoutKind)
	{
	}

	ROIPoolingNode(const ScriptableObjects::IConfigRecordPtr configp)
		: ROIPoolingNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"H"), configp->Get(L"W"),
						 ImageLayoutKindFrom(configp->Get(L"imageLayout")))
	{
		AttachInputsFromConfig(configp, GetExpectedNumInputs());
	}

	// custom forward prop. need to use adaptive pooling window
	// for input ROIs. ROIs are input(0). inputFeatureMaps (infm) are Input(1).
	// ROIs should have dimension [ROI_size, ROIs_per_image, batch_size];
	// we loop over the bsz dimension and depending on the ROI shape use a different
	// pooling window size. TODO: depending on the image shape, need to slice differently into the mb.
	// depends on status of fully conv. for now only works with same-size minibatches.

	void ForwardProp(const FrameRange& fr) override
	{
		//Input(0)->Value().Print(nullptr); // input ROIs
		// reshape as ROI_sz x ROIs/im x mb_sz
		//std::cout << string(Input(0)->GetSampleLayout()).c_str();

		// first dimension is roi_size (4) * rois/image, second is mb size
		int rois_per_image = GetInputSampleLayout(0)[0] / 4;

		//fprintf(stderr, "ROI_PER_IMAGE: %d\n", rois_per_image);

		auto inputImgShape = GetInputSampleLayout(1);
		Matrix<ElemType> inputSlice = Input(1)->ValueFor(fr);
		Matrix<ElemType> ROIs = Input(0)->ValueFor(fr);

		// our output slice for this minibatch.
		// todo: see shape comment in validate.
		Matrix<ElemType> outputSlice = ValueFor(fr);

		// input slice is w*h*c x bsz; cols are images.
		// rois is rois_per_image*4 x bsz; cols are rois for different images.
		// each ROI is (x, y, w, h) relative to original image size.
		// todo: read ROIs, compute from ROI data which features to use in the pooling
		// map and also the kernel size. q: how to slice into inputSlice to get the right features???
		// how can we compute projection from image -> conv2 output?
		//fprintf(stderr, "ROIs NUM COLS: %d, ROWS: %d\n", ROIs.GetNumCols(), ROIs.GetNumRows());
		//fprintf(stderr, "INPUTSLICE NUM COLS: %d, ROWS: %d\n", inputSlice.GetNumCols(), inputSlice.GetNumRows());
		//fprintf(stderr, "INPUT SHAPE %s\n", string(inputImgShape).c_str());

		// fprop loop. looping over images (columns of inputSlice)
		for (int img_idx = 0; img_idx < inputSlice.GetNumCols(); img_idx++) {
			auto img = inputSlice.ColumnSlice(img_idx, 1);
			auto rois = ROIs.ColumnSlice(img_idx, 1);

			//fprintf(stderr, "IMAGE %d ROI COLUMN:\n", img_idx);
			//rois.Print(nullptr);

			// loop over rois per image.
			for (int roi_idx = 0; roi_idx < rois_per_image; roi_idx++) {
				
				// roi points are doubles that represent location relative to image
				int base = roi_idx * 4;
				double x = rois.GetValue(base, 0);
				double y = rois.GetValue(base + 1, 0);
				double w = rois.GetValue(base + 2, 0);
				double h = rois.GetValue(base + 3, 0);
				
				int input_w = inputImgShape[0];
				int input_h = inputImgShape[1];

				// compute actual spatial location of the ROI in our featuremap.
				x = int(x * input_w);
				y = int(y * input_h);

				// make sure rois are at least 1x1.
				size_t roi_w = size_t(max(int(w * input_w), 1));
				size_t roi_h = size_t(max(int(h * input_h), 1));

				// ROI input to ConvolveGeometry pointer...keep the same number of channels
				// as the input image but change the spatial size.
				TensorShape tmp_input_shape = TensorShape(roi_w, roi_h, inputImgShape[2]);
				//img.Reshape()
				//fprintf(stderr, "IMAGE %d ROI %d: (%f %f %f %f)\n", img_idx, roi_idx, x, y, w, h);
				// grab slice of image corresponding to x,y,w,h
				// set up conv geometry / conv engine.

				// can i do this part without having the slicing yet?
				//auto geometry = std::make_shared<ConvolveGeometry>(tmp_input_shape, )
				
			}
		}

		
		/*auto geometry = std::make_shared<ConvolveGeometry>(inputShape, m_kernelShape, m_mapCount, m_stride,
			m_sharing, m_autoPad, m_lowerPad, m_upperPad);

		m_convEng = ConvolutionEngine<ElemType>::Create(geometry, m_deviceId, m_imageLayout,
			m_maxTempMemSizeInSamples, m_poolKind);*/

	}

	void Save(File& fstream) const override
	{
		Base::Save(fstream);
		uint32_t imageLayoutKind = (uint32_t)m_imageLayout;
		fstream << imageLayoutKind << m_outW << m_outH;
	}

	void Load(File& fstream, size_t modelVersion) override
	{
		Base::Load(fstream, modelVersion);
		uint32_t imageLayoutKind;
		fstream >> imageLayoutKind >> m_outW >> m_outH;
		m_imageLayout = (ImageLayoutKind)imageLayoutKind;
	}

	void Validate(bool isFinalValidationPass) override
	{
		Base::Validate(isFinalValidationPass);
		InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

		auto inDims = ImageDimensions(GetInputSampleLayout(1), m_imageLayout);
		fprintf(stderr, "ROI in dims: W: %d, H: %d, C: %d\n", inDims.m_width, inDims.m_height, inDims.m_numChannels);
		
		if (isFinalValidationPass && (inDims.m_width < m_outW || inDims.m_height < m_outH))
			InvalidArgument("ROIPoolingNode: inputWidth must >= windowWidth and inputHeight must >= windowHeight.");

		// todo: this is technically the correct spatial dimension, but we are also increasing the 
		// effective minibatch size to bsz * rois_per_image. so we may need a hack to make that work...
		// not sure how to have different minibatch sizes at different parts of the network in CNTK.
		// need to figure that out if we want to use softmax on top of pooled features rather than SVM.
		auto outDims = ImageDimensions(m_outW, m_outH, inDims.m_numChannels);

		//m_inputSizePerSample = inDims.m_width * inDims.m_height * inDims.m_numChannels;

		SetDims(outDims.AsTensorShape(m_imageLayout), HasMBLayout());

		// don't set geometry yet...need adaptive geometry that depends on ROI shape.
		if (isFinalValidationPass)
		{
			/*// set up various engines and descriptor objects
			m_geometry = std::make_shared<ConvolveGeometry>(inDims.AsTensorShape(m_imageLayoutKind),
				ImageDimensions(m_windowWidth, m_windowHeight, 1).AsTensorShape(m_imageLayoutKind),
				TensorShape(1),
				ImageDimensions(m_horizontalSubsample, m_verticalSubsample, 1).AsTensorShape(m_imageLayoutKind),
				ConvolveGeometry::BoolVec{ true },
				ConvolveGeometry::BoolVec{ false },
				TensorShape(0),
				TensorShape(0));*/
		}
	}

	void BackpropTo(const size_t /*inputIndex*/, const FrameRange& fr) override
	{
		// todo
	}

	void DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const override
	{
		Base::DumpNodeInfo(printValues, printMetadata, fstream);
	}

	void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
	{
		Base::CopyTo(nodeP, newName, flags);
		if (flags & CopyNodeFlags::copyNodeValue)
		{
			auto node = dynamic_pointer_cast<ROIPoolingNode<ElemType>>(nodeP);
			node->m_outW = m_outW;
			node->m_outH = m_outH;
			node->m_imageLayout = m_imageLayout;
		}
	}


protected:
	size_t m_outH, m_outW;
	ImageLayoutKind m_imageLayout; // how to interpret the tensor (which dimensions are X/Y and C)
	ConvolveGeometryPtr m_geometry;
	std::unique_ptr<ConvolutionEngine<ElemType>> m_convEng;

};

// -----------------------------------------------------------------------
// PoolingNode (inputFeature)
// -----------------------------------------------------------------------

template <class ElemType>
class PoolingNode : public ConvolutionNodeBase<ElemType>, public NumInputs<1>
{
    typedef ConvolutionNodeBase<ElemType> Base;
    UsingConvolutionNodeBaseMembers;
    static const std::wstring TypeName()
    {
        return L"Pooling";
    }

public:
    PoolingNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    PoolingNode(DEVICEID_TYPE deviceId, const wstring& name, PoolKind pool, const TensorShape& kernelShape, const TensorShape& strideShape,
                    const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                    ImageLayoutKind imageLayout)
                    : Base(deviceId, name, kernelShape, TensorShape(1), strideShape, vector<bool>{true}, autoPadding, lowerPad, upperPad, pool, imageLayout, 0)
    {
    }
    PoolingNode(const ScriptableObjects::IConfigRecordPtr configp)
        : PoolingNode(configp->Get(L"deviceId"), L"<placeholder>", PoolKindFrom(configp->Get(L"pool")), configp->Get(L"kernelShape"),
                      configp->Get(L"strideShape"),
                      configp->Get(L"dimPadding"), configp->Get(L"dimPadLower"), configp->Get(L"dimPadUpper"),
                      ImageLayoutKindFrom(configp->Get(L"imageLayout")))
    {
        AttachInputsFromConfig(configp, GetExpectedNumInputs());
    }

public:
    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        if (m_imageLayout != ImageLayoutKind::CHW)
        {
            InvalidArgument(
                "%ls %ls supports only cuDNN (CHW) data layout. "
                "Please specify imageLayout=\"cudnn\" in %ls node in your script "
                "and make sure input data layout is CHW", NodeName().c_str(), OperationName().c_str(), NodeName().c_str());
        }

        auto inputShape = GetInputSampleLayout(0);
        auto outDims = ConvolveGeometry::ComputeOutputShape(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                            m_sharing, m_autoPad, m_lowerPad, m_upperPad);
        SetDims(outDims, HasMBLayout());

        if (isFinalValidationPass)
        {
            if (m_convEng == nullptr)
            {
                auto geometry = std::make_shared<ConvolveGeometry>(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                                   m_sharing, m_autoPad, m_lowerPad, m_upperPad);
                m_convEng = ConvolutionEngine<ElemType>::Create(geometry, m_deviceId, m_imageLayout,
                                                                m_maxTempMemSizeInSamples, m_poolKind);
            }
        }
    }
};

// -----------------------------------------------------------------------
// Legacy PoolingNodeBase (input)
// -----------------------------------------------------------------------

template <class ElemType>
class PoolingNodeBase : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembers;

public:
    PoolingNodeBase(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          m_windowWidth(SIZE_MAX),
          m_windowHeight(SIZE_MAX),
          m_horizontalSubsample(SIZE_MAX),
          m_verticalSubsample(SIZE_MAX),
          m_imageLayoutKind(ImageLayoutKind::HWC)
    {
    }
    PoolingNodeBase(DEVICEID_TYPE deviceId, const wstring& name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind)
        : Base(deviceId, name),
          m_windowWidth(windowWidth),
          m_windowHeight(windowHeight),
          m_horizontalSubsample(horizontalSubsample),
          m_verticalSubsample(verticalSubsample),
          m_imageLayoutKind(imageLayoutKind)
    {
    }
    PoolingNodeBase(const ScriptableObjects::IConfigRecordPtr configp)
        : PoolingNodeBase(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"windowWidth"), configp->Get(L"windowHeight"), configp->Get(L"horizontalSubsample"), configp->Get(L"verticalSubsample"), ImageLayoutKindFrom(configp->Get(L"imageLayout")))
    {
        // input, windowWidth, windowHeight, horizontalSubsample, verticalSubsample
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    void Save(File& fstream) const override
    {
        Base::Save(fstream);
        uint32_t imageLayoutKind = (uint32_t) m_imageLayoutKind;
        uint32_t windowWidth = (uint32_t) m_windowWidth;
        fstream << windowWidth << imageLayoutKind << m_windowHeight << m_horizontalSubsample << m_verticalSubsample;
    }

    void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        uint32_t imageLayoutKind, windowWidth;
        fstream >> windowWidth >> imageLayoutKind >> m_windowHeight >> m_horizontalSubsample >> m_verticalSubsample;
        m_windowWidth = windowWidth;
        m_imageLayoutKind = (ImageLayoutKind)imageLayoutKind;
    }

    void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<PoolingNodeBase<ElemType>>(nodeP);

            node->m_windowWidth = m_windowWidth;
            node->m_windowHeight = m_windowHeight;

            node->m_horizontalSubsample = m_horizontalSubsample;
            node->m_verticalSubsample = m_verticalSubsample;

            node->m_inputSizePerSample = m_inputSizePerSample;
            node->m_outputSizePerSample = m_outputSizePerSample;

            node->m_imageLayoutKind = m_imageLayoutKind;
        }
    }

    void BackpropTo(const size_t /*inputIndex*/, const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Grad = Input(0)->GradientFor(fr);
        Matrix<ElemType> sliceOutputGrad = GradientFor(fr);

        Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        m_convEng->BackwardPooling(sliceOutputValue, sliceOutputGrad, sliceInput0Value, sliceInput0Grad);
    }

    void ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        m_convEng->ForwardPooling(sliceInput0Value, sliceOutputValue);
    }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        // get input tensor shape and interpret as image dimensions
        auto inDims = ImageDimensions(GetInputSampleLayout(0), m_imageLayoutKind);

        if (isFinalValidationPass && (inDims.m_width < m_windowWidth || inDims.m_height < m_windowHeight))
            InvalidArgument("PoolingNodeBase: inputWidth must >= windowWidth and inputHeight must >= windowHeight.");

        // determine output tensor shape
        auto outDims = ImageDimensions(
            (inDims.m_width - m_windowWidth) / m_horizontalSubsample + 1,
            (inDims.m_height - m_windowHeight) / m_verticalSubsample + 1,
            inDims.m_numChannels);

        m_inputSizePerSample = inDims.m_width * inDims.m_height * inDims.m_numChannels;

        SetDims(outDims.AsTensorShape(m_imageLayoutKind), HasMBLayout());

        if (isFinalValidationPass)
        {
            // set up various engines and descriptor objects
            m_geometry = std::make_shared<ConvolveGeometry>(inDims.AsTensorShape(m_imageLayoutKind),
                                                            ImageDimensions(m_windowWidth, m_windowHeight, 1).AsTensorShape(m_imageLayoutKind),
                                                            TensorShape(1),
                                                            ImageDimensions(m_horizontalSubsample, m_verticalSubsample, 1).AsTensorShape(m_imageLayoutKind),
                                                            ConvolveGeometry::BoolVec{true},
                                                            ConvolveGeometry::BoolVec{false},
                                                            TensorShape(0),
                                                            TensorShape(0));
        }
    }

    void DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const override
    {
        Base::DumpNodeInfo(printValues, printMetadata, fstream);

        if (printMetadata)
        {
            auto inputSampleLayout = GetInputSampleLayout(0);

            char str[4096];
            sprintf(str, "Input[Width:%lu, Height:%lu, Channels:%lu]  \n", inputSampleLayout[1], inputSampleLayout[2], inputSampleLayout[0]);
            fstream << string(str);
            sprintf(str, "PoolingWindow[Width:%lu, Height:%lu]  SubSampling[Horizontal:%lu, Vertical:%lu]\n", m_windowWidth, m_windowHeight, m_horizontalSubsample, m_verticalSubsample);
            fstream << string(str);
            sprintf(str, "Output[Width:%lu, Height:%lu, Channels:%lu]  \n", m_sampleLayout[1], m_sampleLayout[2], m_sampleLayout[0]);
            fstream << string(str);
            sprintf(str, "TotalSizePerSample[Input:%lu, Output:%lu]  \n", m_inputSizePerSample, m_outputSizePerSample);
            fstream << string(str);
        }
    }

protected:
    size_t m_windowWidth, m_windowHeight;
    size_t m_horizontalSubsample, m_verticalSubsample;
    size_t m_inputSizePerSample, m_outputSizePerSample;

    ImageLayoutKind m_imageLayoutKind; // how to interpret the tensor (which dimensions are X/Y and C)

    ConvolveGeometryPtr m_geometry;
    std::unique_ptr<ConvolutionEngine<ElemType>> m_convEng;
};

// add this at the start of each derived class, to get access to the members of ComputationNode
// See #define of 'UsingComputationNodeMembersBoilerplate' for more explanation.
#define UsingPoolingNodeBaseMembers         \
    UsingComputationNodeMembersBoilerplate; \
    \
protected:                                  \
    using Base::m_geometry;                 \
    using Base::m_convEng;                  \
    using Base::m_windowWidth;              \
    using Base::m_windowHeight;             \
    using Base::m_horizontalSubsample;      \
    using Base::m_verticalSubsample;        \
    using Base::m_inputSizePerSample;       \
    using Base::m_outputSizePerSample;      \
    using Base::m_imageLayoutKind;          \
    \
public:

// -----------------------------------------------------------------------
// Legacy MaxPoolingNode
// -----------------------------------------------------------------------

template <class ElemType>
class MaxPoolingNode : public PoolingNodeBase<ElemType>
{
    typedef PoolingNodeBase<ElemType> Base;
    UsingPoolingNodeBaseMembers;
    static const std::wstring TypeName()
    {
        return L"MaxPooling";
    }

public:
    MaxPoolingNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    MaxPoolingNode(DEVICEID_TYPE deviceId, const wstring& name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind)
        : Base(deviceId, name, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayoutKind)
    {
    }
    MaxPoolingNode(const ScriptableObjects::IConfigRecordPtr configp)
        : Base(configp)
    {
    }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        if (isFinalValidationPass && m_convEng == nullptr)
            m_convEng = ConvolutionEngine<ElemType>::Create(m_geometry, m_deviceId, m_imageLayoutKind, 0, PoolKind::Max);
    }
};

// -----------------------------------------------------------------------
// Legacy AveragePoolingNode
// -----------------------------------------------------------------------

template <class ElemType>
class AveragePoolingNode : public PoolingNodeBase<ElemType>
{
    typedef PoolingNodeBase<ElemType> Base;
    UsingPoolingNodeBaseMembers;
    static const std::wstring TypeName()
    {
        return L"AveragePooling";
    }

public:
    AveragePoolingNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    AveragePoolingNode(DEVICEID_TYPE deviceId, const wstring& name, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind)
        : Base(deviceId, name, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayoutKind)
    {
    }
    AveragePoolingNode(const ScriptableObjects::IConfigRecordPtr configp)
        : Base(configp)
    {
    }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        if (isFinalValidationPass && m_convEng == nullptr)
            m_convEng = ConvolutionEngine<ElemType>::Create(m_geometry, m_deviceId, m_imageLayoutKind, 0, PoolKind::Average);
    }
};

} } }
