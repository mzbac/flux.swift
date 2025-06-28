import Foundation
import MLX
import MLXNN

public enum KontextUtilities {
  
  public static let preferredKontextResolutions: [(Int, Int)] = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672)
  ]
  
  public static func selectKontextResolution(
    width: Int, height: Int
  ) -> (width: Int, height: Int) {
    let aspectRatio = Float(width) / Float(height)
    
    var bestMatch = preferredKontextResolutions[0]
    var minDifference = Float.infinity
    
    for resolution in preferredKontextResolutions {
      let resAspectRatio = Float(resolution.0) / Float(resolution.1)
      let difference = abs(aspectRatio - resAspectRatio)
      
      if difference < minDifference {
        minDifference = difference
        bestMatch = resolution
      }
    }
    
    return (width: bestMatch.0, height: bestMatch.1)
  }
  
  public static func prepareKontextImageIds(
    height: Int, width: Int, batchSize: Int = 1
  ) -> MLXArray {
    let h = height / 16
    let w = width / 16
    
    var imgCondIds = MLX.zeros([h, w, 3])
    
    imgCondIds[0..., 0..., 0] = MLXArray(1)
    
    let heightCoords = MLXArray(0..<h)
    imgCondIds[0..., 0..., 1] = add(
      imgCondIds[0..., 0..., 1], 
      heightCoords[0..., .newAxis]
    )
    
    let widthCoords = MLXArray(0..<w)
    imgCondIds[0..., 0..., 2] = add(
      imgCondIds[0..., 0..., 2], 
      widthCoords[.newAxis, 0...]
    )
    
    imgCondIds = imgCondIds.reshaped(1, h * w, 3)
    
    if batchSize > 1 {
      imgCondIds = MLX.repeated(imgCondIds, count: batchSize, axis: 0)
    }
    
    return imgCondIds
  }
  
  public static func prepareStandardImageIds(
    height: Int, width: Int, batchSize: Int = 1
  ) -> MLXArray {
    let h = height / 16
    let w = width / 16
    
    var imgIds = MLX.zeros([h, w, 3])
    
    let heightCoords = MLXArray(0..<h)
    imgIds[0..., 0..., 1] = add(
      imgIds[0..., 0..., 1], 
      heightCoords[0..., .newAxis]
    )
    
    let widthCoords = MLXArray(0..<w)
    imgIds[0..., 0..., 2] = add(
      imgIds[0..., 0..., 2], 
      widthCoords[.newAxis, 0...]
    )
    
    imgIds = imgIds.reshaped(1, h * w, 3)
    
    if batchSize > 1 {
      imgIds = MLX.repeated(imgIds, count: batchSize, axis: 0)
    }
    
    return imgIds
  }
  
  public static func rearrangeForKontext(_ image: MLXArray) -> MLXArray {
    let shape = image.shape
    let b = shape[0]
    let c = shape[1]
    let h = shape[2]
    let w = shape[3]
    
    precondition(h % 2 == 0 && w % 2 == 0, "Height and width must be divisible by 2")
    
    var rearranged = image.reshaped(b, c, h / 2, 2, w / 2, 2)
    
    rearranged = rearranged.transposed(0, 2, 4, 1, 3, 5)
    
    let result = rearranged.reshaped(b, (h / 2) * (w / 2), c * 4)
    return result
  }
  
  public static func calculateScheduleShift(
    imageSeqLen: Int, 
    baseShift: Float = 0.5, 
    maxShift: Float = 1.15
  ) -> Float {
    let x1: Float = 256
    let y1: Float = 0.5
    let x2: Float = 4096
    let y2: Float = 1.15
    
    let m = (y2 - y1) / (x2 - x1)
    let b = y1 - m * x1
    
    return m * Float(imageSeqLen) + b
  }
  
  public static func timeShift(mu: Float, sigma: Float = 1.0, t: MLXArray) -> MLXArray {
    let expMu = exp(mu)
    let oneOverT = 1.0 / t
    let denominator = pow(oneOverT - 1, sigma)
    return expMu / (expMu + denominator)
  }
  
  public static func resizeImage(_ image: MLXArray, targetWidth: Int, targetHeight: Int) -> MLXArray {
    
    guard image.ndim == 3 else {
      fatalError("resizeImage expects 3D input [H, W, C], got shape: \(image.shape)")
    }
    
    let shape = image.shape
    let currentHeight = shape[0]
    let currentWidth = shape[1]
    let channels = shape[2]
    
    guard channels == 3 else {
      fatalError("Expected 3 channels for RGB image, got \(channels)")
    }
    
    if currentHeight == targetHeight && currentWidth == targetWidth {
      return image
    }
    
    let heightScale = Float(targetHeight) / Float(currentHeight)
    let widthScale = Float(targetWidth) / Float(currentWidth)
    
    // Image is already in HWC format
    let batchedImage = expandedDimensions(image, axis: 0)
    
    if heightScale >= 1.0 && widthScale >= 1.0 {
      let upsampler = Upsample(
        scaleFactor: [heightScale, widthScale],
        mode: .linear(alignCorners: false)
      )
      let resized = upsampler(batchedImage)
      let hwcResult = resized[0, 0..., 0..., 0...]
      return hwcResult
    } else {
      let hIndices = MLXArray(0..<targetHeight).asType(.float32) * Float(currentHeight) / Float(targetHeight)
      let wIndices = MLXArray(0..<targetWidth).asType(.float32) * Float(currentWidth) / Float(targetWidth)
      
      let hIdx = hIndices.asType(.int32)
      let wIdx = wIndices.asType(.int32)
      
      let upsampler = Upsample(
        scaleFactor: [heightScale, widthScale],
        mode: .nearest
      )
      
      let resized = upsampler(batchedImage)
      
      let hwcResult = resized[0, 0..., 0..., 0...]
      return hwcResult
    }
  }
}