//
//  Kokoro.swift
//  mlx-swift-examples
//
//  Created by Cyril Zakka on 03/04/25.
//

import Foundation
import Hub
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Configuration
public struct KokoroConfiguration: Codable, Sendable {
    public let dimIn: Int
    public let dropout: Float
    public let hiddenDim: Int
    public let maxConvDim: Int
    public let maxDur: Int
    public let multispeaker: Bool
    public let nLayer: Int
    public let nMels: Int
    public let nToken: Int
    public let styleDim: Int
    public let textEncoderKernelSize: Int
    
    public let textConfig: TextConfiguration
    public let vocoderConfig: VocoderConfiguration
    public let vocab: [String: Int]
    
    // PL-BERT
    public struct TextConfiguration: Codable, Sendable {
        public let hiddenSize: Int
        public let numAttentionHeads: Int
        public let intermediateSize: Int
        public let maxPositionEmbeddings: Int
        public let numHiddenLayers: Int
        public let dropout: Float
        
        private enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numAttentionHeads = "num_attention_heads"
            case intermediateSize = "intermediate_size"
            case maxPositionEmbeddings = "max_position_embeddings"
            case numHiddenLayers = "num_hidden_layers"
            case dropout
        }
    }
    
    // iSTFTNet
    public struct VocoderConfiguration: Codable, Sendable {
        public let upsampleKernelSizes: [Int]
        public let upsampleRates: [Int]
        public let genIstftHopSize: Int
        public let genIstftNFft: Int
        public let resblockDilationSizes: [[Int]]
        public let resblockKernelSizes: [Int]
        public let upsampleInitialChannel: Int
        
        private enum CodingKeys: String, CodingKey {
            case upsampleKernelSizes = "upsample_kernel_sizes"
            case upsampleRates = "upsample_rates"
            case genIstftHopSize = "gen_istft_hop_size"
            case genIstftNFft = "gen_istft_n_fft"
            case resblockDilationSizes = "resblock_dilation_sizes"
            case resblockKernelSizes = "resblock_kernel_sizes"
            case upsampleInitialChannel = "upsample_initial_channel"
        }
    }

    private enum CodingKeys: String, CodingKey {
        case dimIn = "dim_in"
        case dropout
        case hiddenDim = "hidden_dim"
        case maxConvDim = "max_conv_dim"
        case maxDur = "max_dur"
        case multispeaker
        case nLayer = "n_layer"
        case nMels = "n_mels"
        case nToken = "n_token"
        case styleDim = "style_dim"
        case textEncoderKernelSize = "text_encoder_kernel_size"
        case textConfig = "plbert"
        case vocoderConfig = "istftnet"
        case vocab
    }
    
    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.textConfig = try container.decode(TextConfiguration.self, forKey: .textConfig)
        self.vocoderConfig = try container.decode(VocoderConfiguration.self, forKey: .vocoderConfig)
        self.vocab = try container.decode([String: Int].self, forKey: .vocab)
        
        self.dimIn = try container.decode(Int.self, forKey: .dimIn)
        self.dropout = try container.decode(Float.self, forKey: .dropout)
        self.hiddenDim = try container.decode(Int.self, forKey: .hiddenDim)
        self.maxConvDim = try container.decode(Int.self, forKey: .maxConvDim)
        self.maxDur = try container.decode(Int.self, forKey: .maxDur)
        self.multispeaker = try container.decode(Bool.self, forKey: .multispeaker)
        self.nLayer = try container.decode(Int.self, forKey: .nLayer)
        self.nMels = try container.decode(Int.self, forKey: .nMels)
        self.nToken = try container.decode(Int.self, forKey: .nToken)
        self.styleDim = try container.decode(Int.self, forKey: .styleDim)
        self.textEncoderKernelSize = try container.decode(Int.self, forKey: .textEncoderKernelSize)
    }
}

// MARK: - Vocoder
private enum Decoder {
    fileprivate class WeightedConv1d: Module, UnaryLayer {
        // Properties
        public let weight_g: MLXArray
        public let weight_v: MLXArray
        public let bias: MLXArray?
        public let stride: Int
        public let padding: Int
        public let dilation: Int
        public let groups: Int
        
        /// Initializes a 1D convolution layer with weight normalization
        ///
        /// - Parameters:
        ///   - inputChannels: number of input channels
        ///   - outputChannels: number of output channels
        ///   - kernelSize: size of the convolution filters
        ///   - stride: stride when applying the filter
        ///   - padding: positions to 0-pad the input with
        ///   - dilation: dilation of the convolution
        ///   - groups: number of groups for grouped convolution
        ///   - bias: if `true` add a learnable bias to the output
        ///   - encode: if `true`, bias will have inputChannels size instead of outputChannels
        public init(
            inputChannels: Int,
            outputChannels: Int,
            kernelSize: Int,
            stride: Int = 1,
            padding: Int = 1,
            dilation: Int = 1,
            groups: Int = 1,
            bias: Bool = true,
            encode: Bool = false
        ) {
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            
            // Initialize weight magnitude (g) and direction (v) vectors
            self.weight_g = MLXArray.ones([outputChannels, 1, 1])
            self.weight_v = MLXArray.ones([outputChannels, kernelSize, inputChannels])
            
            // Initialize bias if needed
            if bias {
                self.bias = MLXArray.zeros(encode ? [inputChannels] : [outputChannels])
            } else {
                self.bias = nil
            }
        }
        
        open func callAsFunction(_ x: MLXArray) -> MLXArray {
            let weight = AudioProcessing.weightNorm(weight_v: self.weight_v, weight_g: self.weight_g)
            let reshapedBias = self.bias?.reshaped([1, 1, -1])
            let result: MLXArray
            if x.shape.last == weight.shape.last || self.groups > 1 {
                result = conv1d(
                    x,
                    weight,
                    stride: self.stride,
                    padding: self.padding,
                    dilation: self.dilation,
                    groups: self.groups
                )
            } else {
                result = conv1d(
                    x,
                    weight.transposed(),
                    stride: self.stride,
                    padding: self.padding,
                    dilation: self.dilation,
                    groups: self.groups
                )
            }
            if let bias = reshapedBias {
                return result + bias
            }
            return result
        }
    }
    fileprivate class InstanceNorm: Module {
        // Properties
        public let numFeatures: Int
        public let eps: Float
        public let momentum: Float
        public let affine: Bool
        public let trackRunningStats: Bool
        
        // Optional parameters
        public var weight: MLXArray?
        public var bias: MLXArray?
        public var runningMean: MLXArray?
        public var runningVar: MLXArray?
        
        // Initializes an instance normalization layer
        public init(
            numFeatures: Int,
            eps: Float = 1e-5,
            momentum: Float = 0.1,
            affine: Bool = false,
            trackRunningStats: Bool = false
        ) {
            self.numFeatures = numFeatures
            self.eps = eps
            self.momentum = momentum
            self.affine = affine
            self.trackRunningStats = trackRunningStats
            
            // Initialize parameters
            if self.affine {
                self.weight = MLXArray.ones([numFeatures])
                self.bias = MLXArray.zeros([numFeatures])
            }
            
            if self.trackRunningStats {
                self.runningMean = MLXArray.zeros([numFeatures])
                self.runningVar = MLXArray.ones([numFeatures])
            }
        }
        
        // Check input dimensions - to be implemented by subclasses
        open func checkInputDim(_ input: MLXArray) {
            fatalError("Must be implemented by subclass")
        }
        
        // Get the number of non-batch dimensions - to be implemented by subclasses
        open func getNoBatchDim() -> Int {
            fatalError("Must be implemented by subclass")
        }
        
        // Handle input with no batch dimension
        private func handleNoBatchInput(_ input: MLXArray) -> MLXArray {
            let expanded = MLX.expandedDimensions(input, axis: 0)
            let result = applyInstanceNorm(expanded)
            return MLX.squeezed(result, axis: 0)
        }
        
        // Apply instance normalization
        private func applyInstanceNorm(_ input: MLXArray) -> MLXArray {
            // Get dimensions
            let dims = Array(0..<input.ndim)
            let featureDim = dims[dims.count - getNoBatchDim()]
            
            // Compute statistics along all dims except batch and feature dims
            let reduceDims = dims.filter { $0 != 0 && $0 != featureDim }
            
            var mean: MLXArray
            var variance: MLXArray
            
            if training || !trackRunningStats {
                // Compute mean and variance for normalization
                mean = MLX.mean(input, axes: reduceDims, keepDims: true)
                variance = MLX.variance(input, axes: reduceDims, keepDims: true)
                
                // Update running stats if tracking
                if trackRunningStats && training {
                    // Compute overall mean and variance (across batch too)
                    let overallMean = MLX.mean(mean, axis: 0)
                    let overallVar = MLX.mean(variance, axis: 0)
                    
                    // Update running statistics
                    if let currentMean = runningMean, let currentVar = runningVar {
                        runningMean = (1 - momentum) * currentMean + momentum * overallMean
                        runningVar = (1 - momentum) * currentVar + momentum * overallVar
                    }
                }
            } else {
                // Use running statistics
                var meanShape = Array(repeating: 1, count: input.ndim)
                meanShape[featureDim] = numFeatures
                let varShape = meanShape
                
                mean = MLX.reshaped(runningMean!, meanShape)
                variance = MLX.reshaped(runningVar!, varShape)
            }
            
            // Normalize
            let xNorm = (input - mean) / MLX.sqrt(variance + eps)
            
            // Apply affine transform if needed
            if affine, let weight = weight, let bias = bias {
                var weightShape = Array(repeating: 1, count: input.ndim)
                weightShape[featureDim] = numFeatures
                let biasShape = weightShape
                
                let reshapedWeight = MLX.reshaped(weight, weightShape)
                let reshapedBias = MLX.reshaped(bias, biasShape)
                
                return xNorm * reshapedWeight + reshapedBias
            } else {
                return xNorm
            }
        }
        
        // Forward method (equivalent to __call__ in Python)
        open func callAsFunction(_ input: MLXArray) -> MLXArray {
            checkInputDim(input)
            
            let featureDim = input.ndim - getNoBatchDim()
            if input.shape[featureDim] != numFeatures {
                if affine {
                    fatalError("Expected input's size at dim=\(featureDim) to match numFeatures (\(numFeatures)), but got: \(input.shape[featureDim]).")
                } else {
                    print("Input's size at dim=\(featureDim) does not match numFeatures. You can silence this warning by not passing in numFeatures, which is not used because affine=False")
                }
            }
            
            if input.ndim == getNoBatchDim() {
                return handleNoBatchInput(input)
            }
            
            return applyInstanceNorm(input)
        }
    }
    fileprivate class InstanceNorm1D: InstanceNorm {
        public override init(
            numFeatures: Int,
            eps: Float = 1e-5,
            momentum: Float = 0.1,
            affine: Bool = false,
            trackRunningStats: Bool = false
        ) {
            super.init(
                numFeatures: numFeatures,
                eps: eps,
                momentum: momentum,
                affine: affine,
                trackRunningStats: trackRunningStats
            )
        }
        
        override public func checkInputDim(_ input: MLXArray) {
            guard input.ndim == 2 || input.ndim == 3 else {
                fatalError("Expected 2D or 3D input (batch_size, num_features, length), but got \(input.ndim)D input")
            }
        }
        
        override public func getNoBatchDim() -> Int {
            return 2
        }
    }
    fileprivate class AdaIN1D: Module {
        // Properties
        private let norm: InstanceNorm1D
        private let fc: Linear
        
        // Initializes an Adaptive Instance Normalization layer
        public init(styleDim: Int, numFeatures: Int) {
            self.norm = InstanceNorm1D(numFeatures: numFeatures, affine: false)
            self.fc = Linear(styleDim, numFeatures * 2)
        }
        
        // Forward method
        public func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
            let h = self.fc(s)
            let expanded = MLX.expandedDimensions(h, axis: 2)
            let parts = MLX.split(expanded, parts: 2, axis: 1)
            let gamma = parts[0]
            let beta = parts[1]
            
            // Apply the normalization with learned affine parameters
            let normalized = self.norm(x)
            return (1 + gamma) * normalized + beta
        }
    }
    fileprivate class AdaINResBlock1: Module {
        private let convs1: [Decoder.WeightedConv1d]
        private let convs2: [Decoder.WeightedConv1d]
        private let adain1: [AdaIN1D]
        private let adain2: [AdaIN1D]
        private let alpha1: [MLXArray]
        private let alpha2: [MLXArray]
        
        // Initializes an Adaptive Instance Normalization ResBlock
        public init(
            channels: Int,
            kernelSize: Int = 3,
            dilation: [Int] = [1, 3, 5],
            styleDim: Int = 64
        ) {
            // Initialize convolution layers
            var convs1: [Decoder.WeightedConv1d] = []
            for i in 0..<3 {
                convs1.append(
                    Decoder.WeightedConv1d(
                        inputChannels: channels,
                        outputChannels: channels,
                        kernelSize: kernelSize,
                        stride: 1,
                        padding: AudioProcessing.getPadding(kernel_size: kernelSize, dilation: dilation[i]),
                        dilation: dilation[i]
                    )
                )
            }
            self.convs1 = convs1
            
            // Initialize second set of convolution layers
            var convs2: [Decoder.WeightedConv1d] = []
            for _ in 0..<3 {
                convs2.append(
                    Decoder.WeightedConv1d(
                        inputChannels: channels,
                        outputChannels: channels,
                        kernelSize: kernelSize,
                        stride: 1,
                        padding: AudioProcessing.getPadding(kernel_size: kernelSize, dilation: 1),
                        dilation: 1
                    )
                )
            }
            self.convs2 = convs2
            
            // Initialize AdaIN layers
            var adain1: [AdaIN1D] = []
            var adain2: [AdaIN1D] = []
            for _ in 0..<3 {
                adain1.append(AdaIN1D(styleDim: styleDim, numFeatures: channels))
                adain2.append(AdaIN1D(styleDim: styleDim, numFeatures: channels))
            }
            self.adain1 = adain1
            self.adain2 = adain2
            
            // Initialize alpha parameters
            var alpha1: [MLXArray] = []
            var alpha2: [MLXArray] = []
            for _ in 0..<convs1.count {
                alpha1.append(MLXArray.ones([1, channels, 1]))
                alpha2.append(MLXArray.ones([1, channels, 1]))
            }
            self.alpha1 = alpha1
            self.alpha2 = alpha2
        }
        
        // Snake1D activation function
        private func snake1D(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
            return x + (1 / alpha) * (MLX.sin(alpha * x) ** 2)
        }
        
        // Forward method
        public func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
            var output = x
            
            for i in 0..<self.convs1.count {
                let c1 = self.convs1[i]
                let c2 = self.convs2[i]
                let n1 = self.adain1[i]
                let n2 = self.adain2[i]
                let a1 = self.alpha1[i]
                let a2 = self.alpha2[i]
                
                var xt = n1(output, s)
                xt = snake1D(xt, alpha: a1)
                xt = xt.transposed(axes: [0, 2, 1])  // swapaxes(2, 1)
                xt = c1(xt)
                xt = xt.transposed(axes: [0, 2, 1])  // swapaxes(2, 1)
                xt = n2(xt, s)
                xt = snake1D(xt, alpha: a2)
                xt = xt.transposed(axes: [0, 2, 1])  // swapaxes(2, 1)
                xt = c2(xt)
                xt = xt.transposed(axes: [0, 2, 1])  // swapaxes(2, 1)
                output = xt + output
            }
            
            return output
        }
    }
    fileprivate class MLXSTFT: Module {
        private let filterLength: Int
        private let hopLength: Int
        private let winLength: Int
        private let window: AudioProcessing.WindowType
        
        public init(
            filterLength: Int = 800,
            hopLength: Int = 200,
            winLength: Int = 800,
            window: AudioProcessing.WindowType = .hann
        ) {
            self.filterLength = filterLength
            self.hopLength = hopLength
            self.winLength = winLength
            self.window = window
        }
        
        public func transform(_ inputData: MLXArray) -> (MLXArray, MLXArray) {
            var audioInput = inputData
            if audioInput.ndim == 1 {
                audioInput = MLX.expandedDimensions(audioInput, axis: 0)
            }
            
            var magnitudes: [MLXArray] = []
            var phases: [MLXArray] = []
            
            for batchIdx in 0..<audioInput.shape[0] {
                // Compute STFT
                let stft = AudioProcessing.stft(
                    audioInput[batchIdx],
                    nFft: self.filterLength,
                    hopLength: self.hopLength,
                    winLength: self.winLength,
                    window: self.window,
                    center: true,
                    padMode: .reflect
                )
                
                // Get magnitude
                let magnitude = MLX.abs(stft)
                
                // Get phase
                let phase = AudioProcessing.angle(stft)
                
                magnitudes.append(magnitude)
                phases.append(phase)
            }
            
            // Stack along batch dimension
            let stackedMagnitudes = MLX.stacked(magnitudes, axis: 0)
            let stackedPhases = MLX.stacked(phases, axis: 0)
            
            return (stackedMagnitudes, stackedPhases)
        }
        
        public func inverse(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
            let batchSize = magnitude.shape[0]
            var reconstructed: [MLXArray] = []
            for batchIdx in 0..<batchSize {
                let phaseCont = AudioProcessing.unwrap(phase[batchIdx], axis: 1)
                let real = magnitude[batchIdx] * MLX.cos(phaseCont)
                let imag = magnitude[batchIdx] * MLX.sin(phaseCont)
                let stft = real + imag.asImaginary()
                let audio = AudioProcessing.istft(
                            stft,
                            hopLength: self.hopLength,
                            winLength: self.winLength,
                            window: self.window,
                            center: true,
                            length: nil
                        )
                reconstructed.append(audio)
            }
            let stacked = MLX.stacked(reconstructed, axis: 0)
            return MLX.expandedDimensions(stacked, axes: [1])
        }
        
        public func callAsFunction(_ inputData: MLXArray) -> MLXArray {
            let (magnitude, phase) = self.transform(inputData)
            let reconstruction = self.inverse(magnitude: magnitude, phase: phase)
            return MLX.expandedDimensions(reconstruction, axis: -2)
        }
    }
    
}

