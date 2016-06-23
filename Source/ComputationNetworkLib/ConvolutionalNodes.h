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
        : Base(deviceId, name), m_poolKind(PoolKind::None), m_transpose(false), m_maxTempMemSizeInSamples(0)
    {
    }
    ConvolutionNodeBase(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& kernelShape, const TensorShape& mapCount, const TensorShape& strideShape,
                        const std::vector<bool>& sharing, const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                        PoolKind poolKind, bool transpose, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples)
                        : Base(deviceId, name), m_kernelShape(kernelShape), m_mapCount(mapCount), m_stride(strideShape), m_sharing(sharing),
                        m_autoPad(autoPadding), m_lowerPad(lowerPad), m_upperPad(upperPad), m_poolKind(poolKind), m_transpose(transpose),
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
        fstream << m_transpose;
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
        if (modelVersion >= CNTK_MODEL_VERSION_9)
        {
            fstream >> m_transpose;
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
            node->m_transpose = m_transpose;
            node->m_imageLayout = m_imageLayout;
            node->m_maxTempMemSizeInSamples = m_maxTempMemSizeInSamples;
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
    bool m_transpose;
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
    using Base::m_transpose;                \
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
                    bool transpose, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples)
                    : Base(deviceId, name, kernelShape, mapCount, strideShape, sharing, autoPadding, lowerPad, upperPad, PoolKind::None, transpose, imageLayout, maxTempMemSizeInSamples),
                    m_convolution2D(false)
    {
    }
    ConvolutionNode(DEVICEID_TYPE deviceId, const wstring& name, const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels,
                    const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayout,
                    bool zeroPadding, size_t maxTempMemSizeInSamples)
                    : ConvolutionNode(deviceId, name, TensorShape(kernelWidth, kernelHeight, 1), TensorShape(1, 1, outputChannels),
                                      TensorShape(horizontalSubsample, verticalSubsample, 1), vector<bool>{true},
                                      vector<bool>{zeroPadding}, TensorShape(0), TensorShape(0),
                                      false, imageLayout, maxTempMemSizeInSamples)
    {
        m_convolution2D = true;
    }
    ConvolutionNode(const ScriptableObjects::IConfigRecordPtr configp)
        : ConvolutionNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"kernelShape"), configp->Get(L"mapCount"), configp->Get(L"strideShape"),
                          configp->Get(L"dimSharing"), configp->Get(L"dimPadding"), configp->Get(L"dimPadLower"), configp->Get(L"dimPadUpper"),
                          configp->Get(L"transpose"), ImageLayoutKindFrom(configp->Get(L"imageLayout")), configp->Get(L"maxTempMemSizeInSamples"))
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

    void ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);
        const Matrix<ElemType>& input0 = Input(0)->ValueAsMatrix();
        Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);
        if (!m_transpose)
            m_convEng->Forward(sliceInput1Value, input0, sliceOutputValue, *m_tempMatrix);
        else
        {
            // BackwardData adds results to the output so need to zero them out first.
            // REVIEW alexeyk: should be rolled into BackwardData itself.
            sliceOutputValue.SetValue(0);
            m_convEng->BackwardData(sliceInput1Value, input0, sliceOutputValue, *m_tempMatrix);
        }
    }

    void BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        auto sliceOutputGrad = GradientFor(fr);
        if (inputIndex == 0) // derivative with respect to the weight matrix
        {
            auto& grad = Input(0)->GradientAsMatrix();
            auto sliceInput1Value = Input(1)->ValueFor(fr);
            if (!m_transpose)
                m_convEng->BackwardKernel(sliceOutputGrad, sliceInput1Value, grad, fr.IsAllFrames(), *m_tempMatrix);
            else
                m_convEng->BackwardKernel(sliceInput1Value, sliceOutputGrad, grad, fr.IsAllFrames(), *m_tempMatrix);
        }
        else if (inputIndex == 1) // derivative with respect to the input feature
        {
            auto& input0 = Input(0)->ValueAsMatrix();
            auto sliceInput1Grad = Input(1)->GradientFor(fr);
            if (!m_transpose)
                m_convEng->BackwardData(sliceOutputGrad, input0, sliceInput1Grad, *m_tempMatrix);
            else
            {
                // REVIEW alexeyk: Forward overwrites values in sliceInput1Grad. Should handle correctly instead.
                m_convEng->Forward(sliceOutputGrad, input0, sliceInput1Grad, *m_tempMatrix);
            }
        }
    }

    void Validate(bool isFinalValidationPass) override
    {

        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        size_t inputIdx = GetExpectedNumInputs() - 1;
        TensorShape inputShape;
        TensorShape outputShape;
        // If 2D convolution syntax is used then some of the tensor dimensions need to be inferred.
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

            outputShape = ConvolveGeometry::ComputeOutputShape(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                                m_sharing, m_autoPad, m_lowerPad, m_upperPad);
        }
        else
        {
            inputShape = GetInputSampleLayout(inputIdx);
            if (!m_transpose)
            {
                outputShape = ConvolveGeometry::ComputeOutputShape(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                                    m_sharing, m_autoPad, m_lowerPad, m_upperPad);
            }
            else
            {
                // In case of transpose (deconvolution), node input (inputShape) is really the output of the convolution
                // and node output (outDims) is convolution input. ConvolveGeometry does not care about deconvolutions (it does not have to).
                outputShape = ConvolveGeometry::ComputeInputShape(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                                   m_sharing, m_autoPad, m_lowerPad, m_upperPad);
            }
        }
        // ConvolveGeometry always uses CHW.
        SetDims(ImageDimensions(outputShape, ImageLayoutKind::CHW).AsTensorShape(m_imageLayout), HasMBLayout());

        if (isFinalValidationPass)
        {
            if (m_convEng == nullptr)
            {
                auto geometry = std::make_shared<ConvolveGeometry>(!m_transpose ? inputShape : outputShape,
                                                                   m_kernelShape, m_mapCount, m_stride, 
                                                                   m_sharing, m_autoPad, m_lowerPad, m_upperPad);
                m_convEng = ConvolutionEngine<ElemType>::Create(geometry, m_deviceId, m_imageLayout,
                                                                m_maxTempMemSizeInSamples, m_poolKind,
                                                                ConvolutionEngineKind::All, NodeName());
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
    // Flag that indicates whether the node is created using 2D-syntax.
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

	// use adaptive pooling window
	// for input ROIs. ROIs are input(0). inputFeatureMaps (infm) are Input(1).
	// ROIs should have dimension [ROI_size, ROIs_per_image, batch_size];
	// we loop over the bsz dimension and depending on the ROI shape use a different
	// pooling window size. TODO: depending on the image shape, need to slice differently into the mb.
	// depends on status of fully conv. for now only works with same-size minibatches.

	void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
	{
		Base::RequestMatricesBeforeForwardProp(matrixPool);
		RequestMatrixFromPool(m_tempMatrix, matrixPool);
	}

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

		// outslice is filled in "CHWR" format.
		// outslice(roi_idx,c,h,w) = outslice(roi_idx * roi_size + chan + (row + col*height)*num_chan)
		// outslice should be e.g. rois_per_image*7*7*32 x 32
		// can read ROIs contiguously.
		

		// input slice is w*h*c x bsz; cols are images.
		// rois is rois_per_image*4 x bsz; cols are rois for different images.
		// each ROI is (x, y, w, h) relative to original image size.
		// todo: read ROIs, compute from ROI data which features to use in the pooling
		// map and also the kernel size. q: how to slice into inputSlice to get the right features???
		// how can we compute projection from image -> conv2 output?
		//fprintf(stderr, "ROIs NUM COLS: %d, ROWS: %d\n", ROIs.GetNumCols(), ROIs.GetNumRows());
		//fprintf(stderr, "INPUTSLICE NUM COLS: %d, ROWS: %d\n", inputSlice.GetNumCols(), inputSlice.GetNumRows());
		//fprintf(stderr, "OUTPUT SHAPE: (%lu, %lu)\n", (unsigned long)outputSlice.GetNumRows(), (unsigned long)outputSlice.GetNumCols());

		int input_w = inputImgShape[0];
		int input_h = inputImgShape[1];
		int num_channels = inputImgShape[2];
		int roi_output_size = m_outH*m_outW*num_channels;

		// fprop loop. looping over images (columns of inputSlice)

#pragma omp parallel for
		for (int img_idx = 0; img_idx < inputSlice.GetNumCols(); img_idx++) {
			auto img = inputSlice.ColumnSlice(img_idx, 1);
			auto rois = ROIs.ColumnSlice(img_idx, 1);

			//fprintf(stderr, "IMAGE %d COLUMN:\n", img_idx);
			//img.Print(nullptr);

			// loop over rois per image.
#pragma omp parallel for
			for (int roi_idx = 0; roi_idx < rois_per_image; roi_idx++) {

				int base = roi_idx * 4;

				// scaled ROI numbers (relative to original image size)
				// roi points are doubles that represent location relative to image
				double sc_x = rois.GetValue(base, 0);
				double sc_y = rois.GetValue(base + 1, 0); 
				double sc_w = rois.GetValue(base + 2, 0);
				double sc_h = rois.GetValue(base + 3, 0);

				// compute actual spatial location of the ROI in our featuremap.
				int x = (int)round(sc_x * input_w);
				int y = (int)round(sc_y * input_h);
				int roi_w = (int)max(round(sc_w * input_w), 1.0);
				int roi_h = (int)max(round(sc_h * input_h), 1.0);

				//fprintf(stderr, "ROI %d: (X, Y, W, H) in infm: (%d, %d, %d, %d)\n", roi_idx, x, y, roi_w, roi_h);

				const double winW = double(roi_w) / double(m_outW);
				const double winH = double(roi_h) / double(m_outH);

				// from Ross Girshick fast-rcnn caffe cpu
				// loop over spatial locations in output.
				for (int outw = 0; outw < m_outW; outw++) {
					for (int outh = 0; outh < m_outH; outh++) {
						//fprintf(stderr, "computing output spatial location %d %d\n", outw, outh);

						// compute the top left corner of the input
						// spatial window corresponding to this output unit
						int hstart = (int)floor(double(outh)*winH);
						int wstart = (int)floor(double(outw)*winW);

						// compute bottom right corner (not included)
						int hend = (int)ceil(double(outh + 1) * winH);
						int wend = (int)ceil(double(outw + 1) * winW);

						// offset window based on ROI top left corner.
						// these indices are into the input slice.
						hstart = min(max(hstart + y, 0), input_h); // need - 1 here?
						hend = min(max(hend + y, 0), input_h);
						wstart = min(max(wstart + x, 0), input_w); // need - 1 here?
						wend = min(max(wend + x, 0), input_w);

						//fprintf(stderr, "ROI window: (xmin ymin xmax ymax): (%d %d %d %d)\n", wstart, hstart, wend, hend);

						bool isempty = (hend <= hstart) || (wend <= wstart);

						for (int c = 0; c < num_channels; c++) {
							int output_idx = roi_idx*roi_output_size + c + (outh + outw*m_outH)*num_channels;
							//fprintf(stderr, "going in output location %d\n", output_idx);
							outputSlice(output_idx, img_idx) = -1;

							if (isempty)
								outputSlice(output_idx, img_idx) = 0;

							for (int h = hstart; h < hend; h++) {
								for (int w = wstart; w < wend; w++) {
									int data_idx = c + (h + w*input_h)*num_channels;
									if (img(data_idx, 0) > outputSlice(output_idx, img_idx)) {
										outputSlice(output_idx, img_idx) = img(data_idx, 0);
									}
								}
							}
						}
					}
				}
			}
			/*
			// for debugging. output image slice & roi slice for channel 0.
			if (img_idx == 0) {
				Matrix<ElemType> imgMat = Matrix<ElemType>::Zeros(input_w, input_h, m_deviceId);
				for (int w = 0; w < input_w; w++) {
					for (int h = 0; h < input_h; h++) {
						imgMat(w, h) = img((h + w * input_h) * num_channels, 0);
					}
				}

				Matrix<ElemType> roiMat = Matrix<ElemType>::Zeros(m_outW, m_outH, m_deviceId);
				for (int w = 0; w < m_outW; w++) {
					for (int h = 0; h < m_outH; h++) {
						roiMat(w, h) = outputSlice((h + w * m_outH) * num_channels, 0);
					}
				}

				fprintf(stderr, "IMAGE MAT:\n");
				imgMat.Print(nullptr);
				fprintf(stderr, "ROI MAT:\n");
				roiMat.Print(nullptr);
			}*/
		}
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
		size_t rois_per_image = GetInputSampleLayout(0)[0] / 4;

		if (isFinalValidationPass && m_imageLayout != ImageLayoutKind::CHW)
			InvalidArgument("ROIPoolingNode only supports CHW image layout.");

		fprintf(stderr, "ROI in dims: W: %d, H: %d, C: %d\n", inDims.m_width, inDims.m_height, inDims.m_numChannels);
		
		if (isFinalValidationPass && (inDims.m_width < m_outW || inDims.m_height < m_outH))
			InvalidArgument("ROIPoolingNode: inputWidth must >= windowWidth and inputHeight must >= windowHeight.");

		// todo: this is technically the correct spatial dimension, but we are also increasing the 
		// effective minibatch size to bsz * rois_per_image. so we may need a hack to make that work...
		// not sure how to have different minibatch sizes at different parts of the network in CNTK.
		// need to figure that out if we want to use softmax on top of pooled features rather than SVM.
		//auto outDims = ImageDimensions(m_outW, m_outH, inDims.m_numChannels);

		// hack for now...4D tensor.
		SetDims(TensorShape(m_outW, m_outH, inDims.m_numChannels, rois_per_image), HasMBLayout());
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
	shared_ptr<Matrix<ElemType>> m_tempMatrix;
	std::unique_ptr<ConvolutionEngine<ElemType>> m_convEng;

};

// -----------------------------------------------------------------------
// PoolingNode (inputFeature)
// Performs max or average ND pooling.
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
                : Base(deviceId, name, kernelShape, TensorShape(1), strideShape, vector<bool>{true}, autoPadding, lowerPad, upperPad, pool, false, imageLayout, 0)
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
    void ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);
        const Matrix<ElemType>& input0 = Input(0)->ValueFor(fr);
        m_convEng->ForwardPooling(input0, sliceOutputValue);
    }

    void BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        auto sliceOutputGrad = GradientFor(fr);
        Matrix<ElemType> sliceInput0Grad = Input(0)->GradientFor(fr);
        Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        m_convEng->BackwardPooling(sliceOutputValue, sliceOutputGrad, sliceInput0Value, sliceInput0Grad);
    }

    bool OutputUsedInComputingInputNodesGradients() const override
    {
        // The PoolingNode requires output values only for max pooling.
        return m_poolKind == PoolKind::Max;
    }

    void Validate(bool isFinalValidationPass) override
    {
        auto inputShape = GetInputSampleLayout(0);
        ValidatePooling(inputShape, isFinalValidationPass);
        if (isFinalValidationPass)
        {
            if (m_convEng == nullptr)
            {
                auto geometry = std::make_shared<ConvolveGeometry>(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                                   m_sharing, m_autoPad, m_lowerPad, m_upperPad);
                m_convEng = ConvolutionEngine<ElemType>::Create(geometry, m_deviceId, m_imageLayout,
                                                                m_maxTempMemSizeInSamples, m_poolKind,
                                                                ConvolutionEngineKind::All, NodeName());
            }
        }
    }

protected:
    void ValidatePooling(const TensorShape& inputShape, bool isFinalValidationPass)
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

        auto outDims = ConvolveGeometry::ComputeOutputShape(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                            m_sharing, m_autoPad, m_lowerPad, m_upperPad);
        SetDims(outDims, HasMBLayout());
    }
};

// -----------------------------------------------------------------------
// MaxUnpoolingNode (unpoolInputValues, poolInputValues)
// Performs "max unpooling" operation. Max unpooling mirrors the operation
// performed by max pooling node and depends on the values provided to
// the max pooling node (so unlike deconvolution operation, it is not
// completely independent). Unpooling takes 2 inputs: features to be unpooled,
// which tensor has the same shape as corresponding max pooling node output
// and inputs for the original pooling node. Unpooling node
// produces an output which has the same dimensions as input to the
// corresponding max pooling node (i.e. poolInputValues).
// TODO: need to add support for other pooling types, for example,
// average unpooling. Note that in this case, generic unpooling operation
// will take different number of inputs depending on pooling type.
// -----------------------------------------------------------------------

template <class ElemType>
class MaxUnpoolingNode : public ConvolutionNodeBase<ElemType>, public NumInputs<2>
{
    typedef ConvolutionNodeBase<ElemType> Base;
    UsingConvolutionNodeBaseMembers;
    static const std::wstring TypeName() { return L"MaxUnpooling"; }

public:
    MaxUnpoolingNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }
    MaxUnpoolingNode(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& kernelShape, const TensorShape& strideShape,
                       const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                       ImageLayoutKind imageLayout)
                       : Base(deviceId, name, kernelShape, TensorShape(1), strideShape, vector<bool>{true}, autoPadding, lowerPad, upperPad, PoolKind::Max, true, imageLayout, 0)
    {
    }
    MaxUnpoolingNode(const ScriptableObjects::IConfigRecordPtr configp)
        : MaxUnpoolingNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"kernelShape"),
                           configp->Get(L"strideShape"), configp->Get(L"dimPadding"), configp->Get(L"dimPadLower"), configp->Get(L"dimPadUpper"),
                           ImageLayoutKindFrom(configp->Get(L"imageLayout")))
    {
        AttachInputsFromConfig(configp, GetExpectedNumInputs());
    }

public:
    void ForwardProp(const FrameRange& fr) override
    {
        const Matrix<ElemType>& unpoolInput = Input(0)->ValueFor(fr);
        const Matrix<ElemType>& poolInput = Input(1)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);
        m_convEng->MaxUnpooling(unpoolInput, poolInput, sliceOutputValue);
    }

    void BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        if (inputIndex != 0)
            return;

        auto sliceOutputGrad = GradientFor(fr);
        Matrix<ElemType> sliceInput0Grad = Input(0)->GradientFor(fr);
        // BUGBUG: ForwardPooling overwrites values in sliceInput1Grad. Should handle correctly instead.
        m_convEng->ForwardPooling(sliceOutputGrad, sliceInput0Grad);
    }

    bool OutputUsedInComputingInputNodesGradients() const override { return false; }

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
        // Same as in case of deconvolution, node input (inputShape) is really the output of the max pooling
        // and node output (outDims) is pooling input.
        auto outputShape = ConvolveGeometry::ComputeInputShape(inputShape, m_kernelShape, m_mapCount, m_stride,
                                                               m_sharing, m_autoPad, m_lowerPad, m_upperPad);
        SetDims(outputShape, HasMBLayout());
        if (isFinalValidationPass)
        {
            if (m_convEng == nullptr)
            {
                auto geometry = std::make_shared<ConvolveGeometry>(outputShape, m_kernelShape, m_mapCount, m_stride,
                                                                   m_sharing, m_autoPad, m_lowerPad, m_upperPad);
                // Create reference engine as it's the only engine that implements unpooling.
                m_convEng = ConvolutionEngine<ElemType>::Create(geometry, m_deviceId, m_imageLayout,
                                                                m_maxTempMemSizeInSamples, m_poolKind,
                                                                ConvolutionEngineKind::Reference,
                                                                NodeName());
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
        uint32_t imageLayoutKind = (uint32_t)m_imageLayoutKind;
        uint32_t windowWidth = (uint32_t)m_windowWidth;
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

    void ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        m_convEng->ForwardPooling(sliceInput0Value, sliceOutputValue);
    }

    void BackpropTo(const size_t /*inputIndex*/, const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Grad = Input(0)->GradientFor(fr);
        Matrix<ElemType> sliceOutputGrad = GradientFor(fr);

        Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        m_convEng->BackwardPooling(sliceOutputValue, sliceOutputGrad, sliceInput0Value, sliceInput0Grad);
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
        {
            m_convEng = ConvolutionEngine<ElemType>::Create(m_geometry, m_deviceId, m_imageLayoutKind,
                                                            0, PoolKind::Max,
                                                            ConvolutionEngineKind::All, NodeName());
        }
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
        {
            m_convEng = ConvolutionEngine<ElemType>::Create(m_geometry, m_deviceId, m_imageLayoutKind,
                                                            0, PoolKind::Average, 
                                                            ConvolutionEngineKind::All, NodeName());
        }
    }
};

} } }
