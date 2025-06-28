import Foundation
import MLX
import MLXRandom

public protocol KontextImageToImageGenerator: ImageGenerator, Sendable {
  var transformer: MultiModalDiffusionTransformer { get }
  var vae: VAE { get }
  func conditionText(prompt: String) -> (MLXArray, MLXArray)
  func generateKontextLatents(
    image: MLXArray, 
    parameters: EvaluateParameters
  ) -> KontextDenoiseIterator
}

public struct KontextData {
  public let imgCondSeq: MLXArray
  public let imgCondSeqIds: MLXArray
  public let originalImage: MLXArray
  
  public init(imgCondSeq: MLXArray, imgCondSeqIds: MLXArray, originalImage: MLXArray) {
    self.imgCondSeq = imgCondSeq
    self.imgCondSeqIds = imgCondSeqIds  
    self.originalImage = originalImage
  }
}

extension KontextImageToImageGenerator {
  
  public func generateKontextLatents(
    image: MLXArray,
    parameters: EvaluateParameters
  ) -> KontextDenoiseIterator {
    let shape = image.shape
    
    guard shape.count == 3 else {
      fatalError("Expected 3D image tensor [H, W, C], got shape: \(shape)")
    }
    
    let imageHeight = shape[0]
    let imageWidth = shape[1]
    let (optimalPixelWidth, optimalPixelHeight) = KontextUtilities.selectKontextResolution(
      width: imageWidth, 
      height: imageHeight
    )
    
    let optimalLatentWidth = 2 * (optimalPixelWidth / 16)
    let optimalLatentHeight = 2 * (optimalPixelHeight / 16)
    
    let finalWidth = parameters.width
    let finalHeight = parameters.height
    
    
    let resizedImage = KontextUtilities.resizeImage(
      image, 
      targetWidth: finalWidth, 
      targetHeight: finalHeight
    )
    
    // Image is already in HWC format, just add batch dimension
    let batchedImage = expandedDimensions(resizedImage, axis: 0)  
    let imgCond = vae.encode(batchedImage)
    
    let imgCondNCHW = imgCond.transposed(0, 3, 1, 2)  
    
    let imgCondSeq = KontextUtilities.rearrangeForKontext(imgCondNCHW)
    
    let imgCondSeqIds = KontextUtilities.prepareKontextImageIds(
      height: finalHeight,
      width: finalWidth
    )
    
    let noiseHeight = finalHeight / 16
    let noiseWidth = finalWidth / 16
    let latentSeqLen = noiseHeight * noiseWidth
    
    let noise: MLXArray
    if let seed = parameters.seed {
      MLXRandom.seed(seed)
    }
    
    noise = MLXRandom.normal([1, latentSeqLen, 64])
    
    let (promptEmbeddings, pooledPromptEmbeddings) = conditionText(prompt: parameters.prompt)
    
    let kontextData = KontextData(
      imgCondSeq: imgCondSeq,
      imgCondSeqIds: imgCondSeqIds,
      originalImage: image
    )
    
    var updatedParameters = parameters
    updatedParameters.width = finalWidth
    updatedParameters.height = finalHeight
    
    return KontextDenoiseIterator(
      steps: parameters.numInferenceSteps,
      promptEmbeddings: promptEmbeddings,
      pooledPromptEmbeddings: pooledPromptEmbeddings,
      latents: noise,
      evaluateParameters: updatedParameters,
      transformer: transformer,
      kontextData: kontextData
    )
  }
}

public struct KontextDenoiseIterator: Sequence, IteratorProtocol {
  let steps: Int
  let promptEmbeddings: MLXArray
  let pooledPromptEmbeddings: MLXArray
  var latents: MLXArray
  public var i: Int = 0
  let evaluateParameters: EvaluateParameters
  let transformer: MultiModalDiffusionTransformer
  let kontextData: KontextData
  
  public init(
    steps: Int,
    promptEmbeddings: MLXArray,
    pooledPromptEmbeddings: MLXArray,
    latents: MLXArray,
    evaluateParameters: EvaluateParameters,
    transformer: MultiModalDiffusionTransformer,
    kontextData: KontextData
  ) {
    self.steps = steps
    self.promptEmbeddings = promptEmbeddings
    self.pooledPromptEmbeddings = pooledPromptEmbeddings
    self.latents = latents
    self.evaluateParameters = evaluateParameters
    self.transformer = transformer
    self.kontextData = kontextData
  }
  
  public mutating func next() -> MLXArray? {
    guard i < steps else {
      return nil
    }
    
    let imgInput = MLX.concatenated([latents, kontextData.imgCondSeq], axis: 1)
    
    let noise = transformer(
      t: i,
      promptEmbeds: promptEmbeddings,
      pooledPromptEmbeds: pooledPromptEmbeddings,
      hiddenStates: imgInput,
      evaluateParameters: evaluateParameters,
      imgIds: kontextData.imgCondSeqIds  
    )
    
    let generationNoise = noise[0..., 0..<latents.dim(1), 0...]
    
    let dt = evaluateParameters.sigmas[i + 1] - evaluateParameters.sigmas[i]
    latents += generationNoise * dt
    
    i += 1
    return latents
  }
}

extension EvaluateParameters {
  
  public static func createKontextSigmas(
    numInferenceSteps: Int,
    imageSeqLen: Int,
    width: Int,
    height: Int
  ) -> MLXArray {
    var sigmas = MLXArray.linspace(1, 1.0 / Float(numInferenceSteps), count: numInferenceSteps)
    
    let mu = KontextUtilities.calculateScheduleShift(imageSeqLen: imageSeqLen)
    
    sigmas = KontextUtilities.timeShift(mu: mu, t: sigmas)
    
    sigmas = MLX.concatenated([sigmas, MLXArray.zeros([1])])
    
    return sigmas
  }
}