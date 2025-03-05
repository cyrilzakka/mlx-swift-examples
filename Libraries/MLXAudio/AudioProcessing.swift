//
//  File.swift
//  mlx-libraries
//
//  Created by Cyril Zakka on 3/4/25.
//

import MLX
import MLXLMCommon
import Accelerate
import MLXFFT


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

// MARK: STFT and iSTFT
extension AudioProcessing {
    public enum WindowType: String {
        case hann = "hann"
    }
    
    public enum PaddingMode {
        case constant
        case edge
        case reflect
        
        var mlxMode: MLX.PadMode? {
            switch self {
            case .constant:
                return .constant
            case .edge:
                return .edge
            case .reflect:
                return nil
            }
        }
    }
    
    static private func pad(_ x: MLXArray, padding: Int, padMode: PaddingMode) -> MLXArray {
         switch padMode {
         case .constant:
             return MLX.padded(x, width: [padding, padding], mode: .constant)
         case .edge:
             return MLX.padded(x, width: [padding, padding], mode: .edge)
         case .reflect:
             // TODO: if buggy final result, check here second
             let prefix = x[1 ..< (padding + 1)][.stride(by: -1)]
            let suffix = x[-(padding + 1) ..< -1][.stride(by: -1)]
             return concatenated([prefix, x, suffix], axis: 0)
         }
     }
    
    static public func stft(
        _ x: MLXArray,
        nFft: Int = 800,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        window: WindowType = .hann,
        center: Bool = true,
        padMode: PaddingMode = .reflect
    ) -> MLXArray {
        let actualHopLength = hopLength ?? nFft / 4
        let actualWinLength = winLength ?? nFft
        var w: MLXArray
        
        switch window {
        case .hann:
            
            var hannWindow = [Float](repeating: 0, count: actualWinLength)
            vDSP_hann_window(&hannWindow, vDSP_Length(actualWinLength), Int32(0))
            w = MLXArray(hannWindow)
        }
        
        var paddedWindow = w
        if w.shape[0] < nFft {
            let padSize = nFft - w.shape[0]
            paddedWindow = MLX.concatenated([w, MLXArray.zeros([padSize])], axis: 0)
        }

        var inputX = x
        if center {
            inputX = pad(inputX, padding: nFft / 2, padMode: padMode)
        }
        let numFrames = 1 + (inputX.shape[0] - nFft) / actualHopLength
        if numFrames <= 0 {
            fatalError("Input is too short (length=\(inputX.shape[0])) for nFft=\(nFft) with hopLength=\(actualHopLength) and center=\(center).")
        }
        let shape = [numFrames, nFft]
        let strides = [actualHopLength, 1]
        let frames = MLX.asStrided(inputX, shape, strides: strides)
        let spec = MLXFFT.rfft(frames * paddedWindow)
        return transposed(spec, axes: [1, 0])
    }
    
    static public func istft(
        _ x: MLXArray,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        window: WindowType = .hann,
        center: Bool = true,
        length: Int? = nil
    ) -> MLXArray {
        let actualWinLength = winLength ?? (x.shape[1] - 1) * 2
        let actualHopLength = hopLength ?? actualWinLength / 4
        
        // Create window function
        var w: MLXArray
        switch window {
        case .hann:
            var hannWindow = [Float](repeating: 0, count: actualWinLength)
            vDSP_hann_window(&hannWindow, vDSP_Length(actualWinLength), Int32(0))
            w = MLXArray(hannWindow)
        }
        
        // Pad window if needed
        if w.shape[0] < actualWinLength {
            w = MLX.concatenated([w, MLXArray.zeros([actualWinLength - w.shape[0]])], axis: 0)
        }

        let transposedX = transposed(x, axes: [1, 0])
        let t = (transposedX.shape[0] - 1) * actualHopLength + actualWinLength
        var reconstructed = MLXArray.zeros([t])
        let windowSum = MLXArray.zeros([t])
        for i in 0..<transposedX.shape[0] {
            let frameTime = MLXFFT.irfft(transposedX[i])
            let start = i * actualHopLength
            let end = start + actualWinLength

            let timeSlice = start..<end
            reconstructed[timeSlice] += frameTime * w
            windowSum[timeSlice] += w ** 2  // Square of window
        }
        reconstructed = MLX.`where`(windowSum .!= 0, reconstructed / windowSum, reconstructed)
        if center && length == nil {
            reconstructed = reconstructed[actualWinLength / 2 ..< -actualWinLength / 2]
        }
        if let length = length {
            reconstructed = reconstructed[..<length]
        }
        
        return reconstructed
    }
    
    // Converted from https://github.com/numpy/numpy/blob/v2.2.0/numpy/lib/_function_base_impl.py#L1649-L1700
    static public func angle(_ z: MLXArray, deg: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        let real: MLXArray
        let imag: MLXArray
        
        if z.dtype == .complex64 {
            real = z.realPart()
            imag = z.imaginaryPart()
        } else {
            real = z
            imag = MLX.zeros(like: z)
        }
        
        var result = MLX.atan2(imag, real, stream: stream)
        if deg {
            result = result * (180.0 / .pi)
        }
        
        return result
    }
    
    /// Source: https://github.com/ml-explore/mlx/blob/fd0d63ba5b83da41d6a0e75a0bcb5d70ec93eb40/mlx/utils.cpp#L117
    static func normalizeAxisIndex(
        axis: Int,
        ndim: Int,
        msgPrefix: String = ""
    ) -> Int {
        if axis < -ndim || axis >= ndim {
            let message = "\(msgPrefix)Axis \(axis) is out of bounds for array with \(ndim) dimensions."
           fatalError(message)
        }
        return axis < 0 ? axis + ndim : axis
    }
    
    /// Source https://github.com/numpy/numpy/blob/2f7fe64b8b6d7591dd208942f1cc74473d5db4cb/numpy/lib/_function_base_impl.py#L1387
    static public func diff(_ a: MLXArray, n: Int = 1, axis: Int = -1, prepend: MLXArray? = nil, append: MLXArray? = nil) -> MLXArray {
        if n == 0 {
            return a
        }
        guard n >= 0 else {
            fatalError("Order must be non-negative (got \(n))")
        }
        guard a.ndim > 0 else {
            fatalError("diff expects at least one dimension")
        }
        let normAxis = normalizeAxisIndex(axis: axis, ndim: a.ndim)
        
        // Handle prepend and append
        var result = a
        var combined: [MLXArray] = []
        
        if let prepend {
            var prependArray = prepend
            if prependArray.ndim == 0 {
                var shape = Array(repeating: 1, count: a.ndim)
                for i in 0..<a.ndim {
                    if i != normAxis {
                        shape[i] = a.shape[i]
                    }
                }
                prependArray = broadcast(prependArray, to: shape)
            }
            
            combined.append(prependArray)
        }
        
        combined.append(a)
        
        if let append {
            var appendArray = append
            
            // Broadcast scalar to correct shape if needed
            if appendArray.ndim == 0 {
                var shape = Array(repeating: 1, count: a.ndim)
                for i in 0..<a.ndim {
                    if i != normAxis {
                        shape[i] = a.shape[i]
                    }
                }
                appendArray = broadcast(appendArray, to: shape)
            }
            
            combined.append(appendArray)
        }
        
        // Concatenate if needed
        if combined.count > 1 {
            result = concatenated(combined, axis: normAxis)
        }
        
        for _ in 0..<n {
            let firstSlice = result[normAxis < 0 ? (normAxis+1)... : (1...)]
            let secondSlice = result[normAxis < 0 ? ...normAxis : ...(result.shape[normAxis]-2)]
            result = subtract(firstSlice, secondSlice)
        }
        
        return result
    }

    
    /// Converted from https://github.com/numpy/numpy/blob/v2.1.0/numpy/lib/_function_base_impl.py#L1731-L1825
    static public func unwrap(_ p: MLXArray,
                      discont: Float? = nil,
                      axis: Int = -1,
                      period: Float = 2 * Float.pi,
                      stream: StreamOrDevice = .default) -> MLXArray {
        let nd = p.ndim
        let normAxis = normalizeAxisIndex(axis: axis, ndim: nd)
        let dd = diff(p, n: 1, axis: normAxis)
        let discontinuity = discont ?? period/2
        
        let dtype = p.dtype
        let isIntegerType = dtype == .int8 || dtype == .int16 || dtype == .int32 || dtype == .int64 ||
        dtype == .uint8 || dtype == .uint16 || dtype == .uint32 || dtype == .uint64
        
        var isBoundaryAmbiguous: Bool = false
        var intervHigh: MLXArray = []
        if isIntegerType {
            let (intervalHigh, rem) = divmod(dd, discontinuity)
            intervHigh = intervalHigh
            isBoundaryAmbiguous = rem.item() == 0
        } else {
            intervHigh = MLXArray(period / 2)
            isBoundaryAmbiguous = true
        }
        let intervalLow = -intervHigh
        var ddMod = divmod(dd - intervalLow, period).1 + intervalLow
        if isBoundaryAmbiguous {
            let equalMask = MLX.equal(ddMod, intervalLow)
            let greaterMask = MLX.greater(dd, 0)
            let mask = MLX.logicalAnd(equalMask, greaterMask)
            ddMod = MLX.where(mask, intervHigh, ddMod)
        }
        var phCorrect = ddMod - dd
        let lessThanMask = MLX.less(abs(dd), discontinuity)
        phCorrect = MLX.where(lessThanMask, 0, phCorrect)
        
        let up = MLXArray(data: p.asData())
        let cumulativeCorrections = cumsum(phCorrect, axis: normAxis)
        var indices = Array(repeating: 0..., count: nd)
        indices[normAxis] = 1...
        up[indices] = p[indices] + cumulativeCorrections
        
        return up
    }
}
