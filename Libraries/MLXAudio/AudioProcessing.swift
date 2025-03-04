//
//  File.swift
//  mlx-libraries
//
//  Created by Cyril Zakka on 3/4/25.
//

import MLX
import MLXLMCommon


public enum AudioProcessing {}

// MARK: Interpolation
extension AudioProcessing {
    public enum AudioInterpolationMode: String {
        case nearest = "nearest"
        case linear = "linear"
    }
    
    static public func interpolate(input: MLXArray, size: Int, mode: AudioInterpolationMode = .linear, align_corners: Bool = false) -> MLXArray {
        let ndim = input.ndim
        guard ndim >= 3 else {
            fatalError("Expected at least 3D input (N, C, D1), got \(ndim)D")
        }
        let spatial_dims = ndim - 2
        guard spatial_dims == 1 else {
            fatalError("Only 1D interpolation currently supported. Got \(spatial_dims)")
        }
        let actualSize = Array(repeating: size, count: spatial_dims)
        return interpolate1D(input: input, size: actualSize[0], mode: mode, alignCorners: align_corners)
    }
    
    static public func interpolate(input: MLXArray, size: [Int], mode: AudioInterpolationMode = .linear, align_corners: Bool = false) -> MLXArray {
        return interpolate1D(input: input, size: size[0], mode: mode, alignCorners: align_corners)
    }
    
    static public func interpolate(input: MLXArray, scale_factor: Float, mode: AudioInterpolationMode = .linear, align_corners: Bool = false) -> MLXArray {
        let ndim = input.ndim
        guard ndim >= 3 else {
            fatalError("Expected at least 3D input (N, C, D1), got \(ndim)D")
        }
        let spatial_dims = ndim - 2
        guard spatial_dims == 1 else {
            fatalError("Only 1D interpolation currently supported. Got \(spatial_dims)")
        }
        let actualScaleFactor = Array(repeating: scale_factor, count: spatial_dims)
        
        var sizes: [Int] = []
        for i in (0..<spatial_dims) {
            let floatValue = Float(input.shape[i+2]) * actualScaleFactor[i]
            let curr_size = max(1, Int(floatValue.rounded(.up)))
            sizes.append(curr_size)
        }
        return interpolate1D(input: input, size: sizes[0], mode: mode, alignCorners: align_corners)
    }
    
    static public func interpolate(input: MLXArray, scale_factor: [Float], mode: AudioInterpolationMode = .linear, align_corners: Bool = false) -> MLXArray {
        let ndim = input.ndim
        guard ndim >= 3 else {
            fatalError("Expected at least 3D input (N, C, D1), got \(ndim)D")
        }
        let spatial_dims = ndim - 2
        guard spatial_dims == 1 else {
            fatalError("Only 1D interpolation currently supported. Got \(spatial_dims)")
        }
        var sizes: [Int] = []
        for i in (0..<spatial_dims) {
            let floatValue = Float(input.shape[i+2]) * scale_factor[i]
            let curr_size = max(1, Int(floatValue.rounded(.up)))
            sizes.append(curr_size)
        }
        return interpolate1D(input: input, size: sizes[0], mode: mode, alignCorners: align_corners)
    }

    
    // MARK: Private methods
    static private func interpolate1D(input: MLXArray, size: Int, mode: AudioInterpolationMode = .linear, alignCorners: Bool = false) -> MLXArray {
        let (batchSize, channels, inWidth) = (input.shape[0], input.shape[1], input.shape[2])
        
        let actualSize = size < 1 ? 1 : size
        let actualInWidth = inWidth < 1 ? 1 : inWidth
        
        switch mode {
        case .nearest:
            if actualSize == 1 {
                let indices = MLXArray([0])
                return input[.ellipsis, indices]
            } else {
                let scale = Float(actualInWidth) / Float(actualSize)
                let indices = MLXArray(Array(0 ..< actualSize)) * scale
                let flooredIndices = MLX.floor(indices).asType(.int32)
                let clippedIndices = MLX.clip(flooredIndices, min: 0, max: actualInWidth - 1)
                return input[.ellipsis, clippedIndices]
            }
        case .linear:
            var x: MLXArray
            if alignCorners && actualSize > 1 {
                x = MLXArray(Array(0 ..< actualSize)) * ((Float(actualInWidth) - 1) / (Float(actualSize) - 1))
            } else {
                if actualSize == 1 {
                    x = MLXArray(converting: [0.0])
                } else {
                    x = MLXArray(Array(0 ..< actualSize)) * (Float(actualInWidth) / Float(actualSize))
                    if alignCorners != true {
                        x = x + 0.5 * (Float(actualInWidth) / Float(actualSize)) - 0.5
                    }
                }
            }
            
            if actualInWidth == 1 {
                let newShape = [batchSize, channels, actualSize]
                let output = MLX.broadcast(input, to: newShape)
                return output
            }
            
            let xLow = MLX.floor(x).asType(.int32)
            let xHigh = MLX.minimum(xLow + 1, actualInWidth - 1)
            let xFrac = x - xLow
            
            // Pre-compute indices to avoid repeated computation
            let yLow = input[.ellipsis, xLow]
            let yHigh = input[.ellipsis, xHigh]
            
            // Vectorized interpolation
            let oneMinusXFrac = 1 - xFrac
            
            // Add dimensions to xFrac and oneMinusXFrac to match the input dimensions
            let expandedXFrac = xFrac.reshaped([1, 1, actualSize])
            let expandedOneMinusXFrac = oneMinusXFrac.reshaped([1, 1, actualSize])
            
            let output = yLow * expandedOneMinusXFrac + yHigh * expandedXFrac
            return output
        }
    }
}

// MARK: Vocoder helpers
extension AudioProcessing {
    
    public enum NormalizationType: String {
        case l1 = "L1"
        case l2 = "L2"
    }
    
    static func getPadding(kernel_size: Int, dilation: Int = 1) -> Int {
        return Int((kernel_size * dilation - dilation) / 2)
    }
    
    static func computeNorm(x: MLXArray, p: NormalizationType, dim: Int, keepDim: Bool = false) -> MLXArray {
        let dimensions = [dim]
        return computeNorm(x: x, p: p, dim: dimensions)
    }
    
    static func computeNorm(x: MLXArray, p: NormalizationType, keepDim: Bool = false) -> MLXArray {
    let dimensions = Array(0..<x.ndim)
    return computeNorm(x: x, p: p, dim: dimensions)
}
    
    static func computeNorm(x: MLXArray, p: NormalizationType, dim: [Int], keepDim: Bool = false) -> MLXArray {
        switch p {
        case .l1:
            return MLX.sum(x, axes: dim, keepDims: keepDim)
        case .l2:
            return MLX.sqrt(MLX.sum(x * x, axes: dim, keepDims: keepDim))
        }
    }
    
    static public func weightNorm(weight_v: MLXArray, weight_g: MLXArray, dim: Int? = nil) -> MLXArray {
        let rank = weight_v.shape.count
        var axes = Array(0..<rank)
        if let dim {
            var modDim = dim
            if dim < -1 {
                modDim += rank
            }
            if modDim != -1 {
                axes.remove(at: modDim)
            }
        }
        let norm_v = computeNorm(x: weight_v, p: .l2, dim: axes, keepDim: true)
        let normalized_weight = weight_v / (norm_v + 1e-7)
        return normalized_weight * weight_g
    }
    
}
