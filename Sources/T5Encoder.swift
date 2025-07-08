import Foundation
import MLX
import MLXFast
import MLXNN
import Logging

private let logger = Logger(label: "flux.swift.T5Encoder")

public struct T5Configuration {
  var vocabSize = 32128
  var dModel = 4096
  var dKv = 64
  var dFf = 10240
  var numHeads = 64
  var numLayers = 24
  var layerNormEpsilon: Float = 1e-6
  var relativeAttentionNumBuckets = 32
  var relativeAttentionMaxDistance = 128

  public init(
    vocabSize: Int = 32128,
    dModel: Int = 4096,
    dKv: Int = 64,
    dFf: Int = 10240,
    numHeads: Int = 64,
    numLayers: Int = 24,
    layerNormEpsilon: Float = 1e-6,
    relativeAttentionNumBuckets: Int = 32,
    relativeAttentionMaxDistance: Int = 128
  ) {
    self.vocabSize = vocabSize
    self.dModel = dModel
    self.dKv = dKv
    self.dFf = dFf
    self.numHeads = numHeads
    self.numLayers = numLayers
    self.layerNormEpsilon = layerNormEpsilon
    self.relativeAttentionNumBuckets = relativeAttentionNumBuckets
    self.relativeAttentionMaxDistance = relativeAttentionMaxDistance
  }
}

public class MultiHeadAttention: Module {
  @ModuleInfo var q: Linear
  @ModuleInfo var k: Linear
  @ModuleInfo var v: Linear
  @ModuleInfo var o: Linear
  let numHeads: Int
  let dKv: Int

  init(_ config: T5Configuration) {
    let innerDim = config.dKv * config.numHeads
    self.numHeads = config.numHeads
    self.dKv = config.dKv
    self._q.wrappedValue = Linear(config.dModel, innerDim, bias: false)
    self._k.wrappedValue = Linear(config.dModel, innerDim, bias: false)
    self._v.wrappedValue = Linear(config.dModel, innerDim, bias: false)
    self._o.wrappedValue = Linear(innerDim, config.dModel, bias: false)

  }

  func callAsFunction(_ hiddenStates: MLXArray, positionBias: MLXArray) -> MLXArray {
    let queryStates = MultiHeadAttention.shape(q(hiddenStates), numHeads: numHeads, dKv: dKv)
    let keyStates = MultiHeadAttention.shape(k(hiddenStates), numHeads: numHeads, dKv: dKv)
    let valueStates = MultiHeadAttention.shape(v(hiddenStates), numHeads: numHeads, dKv: dKv)

    var scores = MLX.matmul(queryStates, keyStates.transposed(0, 1, 3, 2))

    scores = scores + positionBias.asType(scores.dtype)

    let attentionWeights = MLX.softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
    var attnOutput = MLX.matmul(attentionWeights, valueStates)
    attnOutput = MultiHeadAttention.unShape(attnOutput)
    attnOutput = o(attnOutput)

    return attnOutput
  }

  static func shape(_ states: MLXArray, numHeads: Int, dKv: Int) -> MLXArray {
    states.reshaped(1, -1, numHeads, dKv).transposed(0, 2, 1, 3)
  }

  static func unShape(_ states: MLXArray) -> MLXArray {
    states.transposed(0, 2, 1, 3).reshaped(1, -1, states.dim(1) * states.dim(3))
  }

}

public class T5Attention: Module {
  @ModuleInfo(key: "layer_norm") var layerNorm: RMSNorm
  @ModuleInfo(key: "SelfAttention") var attention: MultiHeadAttention

  init(_ config: T5Configuration) {
    self._layerNorm.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
    self._attention.wrappedValue = MultiHeadAttention(config)
  }

  public func callAsFunction(_ hiddenStates: MLXArray, positionBias: MLXArray) -> MLXArray {
    var x = layerNorm(hiddenStates)
    x = attention(x, positionBias: positionBias)
    return x + hiddenStates
  }
}
public class DenseActivation: Module {
  @ModuleInfo(key: "wi_0") var wi0: Linear
  @ModuleInfo(key: "wi_1") var wi1: Linear
  @ModuleInfo var wo: Linear

  init(_ config: T5Configuration) {
    let inputDim = config.dModel
    let hiddenDim = config.dFf
    self._wi0.wrappedValue = Linear(inputDim, hiddenDim, bias: false)
    self._wi1.wrappedValue = Linear(inputDim, hiddenDim, bias: false)
    self._wo.wrappedValue = Linear(hiddenDim, inputDim, bias: false)
  }

  func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    let hiddenGelu = geluApproximate(wi0(hiddenStates))
    let hiddenLinear = wi1(hiddenStates)
    var hiddenStates = hiddenGelu * hiddenLinear
    hiddenStates = wo(hiddenStates)
    return hiddenStates
  }
}

public class T5FeedForward: Module {
  @ModuleInfo(key: "DenseReluDense") var denseReluDense: DenseActivation
  @ModuleInfo(key: "layer_norm") var layerNorm: RMSNorm
  init(_ config: T5Configuration) {
    self._denseReluDense.wrappedValue = DenseActivation(config)
    self._layerNorm.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
  }

  public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    var x = layerNorm(hiddenStates)
    x = denseReluDense(x)
    return x + hiddenStates
  }
}

public class TransformerEncoderLayer: Module {
  let layer: [Module]

  init(_ config: T5Configuration) {
    self.layer = [T5Attention(config), T5FeedForward(config)]
  }

  func callAsFunction(_ x: MLXArray, positionBias: MLXArray) -> MLXArray {
    var hiddenStates = x
    for module in layer {
      if let attention = module as? T5Attention {
        hiddenStates = attention(hiddenStates, positionBias: positionBias)
      } else if let feedForward = module as? T5FeedForward {
        hiddenStates = feedForward(hiddenStates)
      }
    }
    return hiddenStates
  }
}

public class TransformerEncoder: Module {
  let block: [TransformerEncoderLayer]
  @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: RMSNorm

  init(_ config: T5Configuration) {
    self.block = (0..<config.numLayers).map { _ in TransformerEncoderLayer(config) }
    self._finalLayerNorm.wrappedValue = RMSNorm(
      dimensions: config.dModel, eps: config.layerNormEpsilon)
  }

  func callAsFunction(_ hiddenStates: MLXArray, positionBias: MLXArray) -> MLXArray {
    var x = hiddenStates
    for layer in block {
      x = layer(x, positionBias: positionBias)
    }
    x = finalLayerNorm(x)

    return x
  }
}

public class T5Encoder: Module {
  @ModuleInfo var shared: Embedding
  let encoder: TransformerEncoder
  @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: Embedding

  public init(_ config: T5Configuration) {
    self._shared.wrappedValue = Embedding(
      embeddingCount: config.vocabSize, dimensions: config.dModel)
    self.encoder = TransformerEncoder(config)
    self._relativeAttentionBias.wrappedValue = Embedding(
      embeddingCount: config.relativeAttentionNumBuckets, dimensions: config.numHeads)
  }

  public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
    let positionBias = computeBias(seqLength: inputIds.dim(1))

    let hiddenStates = shared(inputIds)
    let encoderOutputs = encoder(hiddenStates, positionBias: positionBias)
    return encoderOutputs
  }

  func computeBias(seqLength: Int) -> MLXArray {
    let contextPosition = MLXArray(0..<seqLength)[0..., .newAxis]
    let memoryPosition = MLXArray(0..<seqLength)[.newAxis, 0...]
    let relativePosition = memoryPosition - contextPosition
    let relativePositionBucket = T5Encoder.relativePositionBucket(
      relativePosition: relativePosition)

    var values = relativeAttentionBias(relativePositionBucket)
    values = values.transposed(2, 0, 1)
    values = MLX.expandedDimensions(values, axis: 0)
    return values
  }

  static func relativePositionBucket(
    relativePosition: MLXArray, bidirectional: Bool = true, numBuckets: Int = 32,
    maxDistance: Int = 128
  ) -> MLXArray {
    var relativeBuckets = MLXArray.zeros(relativePosition.shape, dtype: .int32)
    var numBucketsVar = numBuckets

    numBucketsVar /= 2
    relativeBuckets =
      relativeBuckets + MLX.where(relativePosition .> 0, MLXArray(numBucketsVar), MLXArray(0))
    let relativePositionAbs = MLX.abs(relativePosition)

    let maxExact = numBucketsVar / 2
    let isSmall = relativePositionAbs .< maxExact

    let scale = Float(numBucketsVar - maxExact) / log(Float(maxDistance) / Float(maxExact))
    let relativePositionIfLarge =
      maxExact
      + (MLX.log(relativePositionAbs.asType(.float32) / Float(maxExact)) * scale).asType(.int32)
    let relativePositionIfLargeClamped = MLX.minimum(
      relativePositionIfLarge, MLXArray(numBucketsVar - 1))

    relativeBuckets =
      relativeBuckets + MLX.where(isSmall, relativePositionAbs, relativePositionIfLargeClamped)
    return relativeBuckets
  }
}