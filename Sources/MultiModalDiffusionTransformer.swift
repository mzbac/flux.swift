import Foundation
import MLX
import MLXFast
import MLXNN

public struct MultiModalDiffusionConfiguration {
  public var attentionHeadDim = 128
  public var guidanceEmbeds = false
  public var inChannels = 64
  public var jointAttentionDim = 4096
  public var numAttentionHeads = 24
  public var numLayers = 19
  public var numSingleLayers = 38
  public var patchSize = 1
  public var pooledProjectionDim = 768
  public var axesDimsRope: (Int, Int, Int) = (16, 56, 56)
  public var layerNormEps: Float = 1e-6

  public init(guidanceEmbeds: Bool = false) {
    self.guidanceEmbeds = guidanceEmbeds
  }
}

public class EmbedND: Module {
  let dim: Int
  let theta: Float
  let axesDim: [Int]

  init(_ config: MultiModalDiffusionConfiguration) {
    self.dim = config.numAttentionHeads * config.attentionHeadDim
    self.theta = 10000
    self.axesDim = [config.axesDimsRope.0, config.axesDimsRope.1, config.axesDimsRope.2]
  }

  func callAsFunction(_ ids: MLXArray) -> MLXArray {
    let emb = MLX.concatenated(
      (0..<3).map { i in
        let slice = ids[.ellipsis, i]
        let ropeResult = EmbedND.rope(slice, dim: axesDim[i], theta: theta)
        return ropeResult
      },
      axis: -3
    )
    
    return MLX.expandedDimensions(emb, axis: 1)
  }

  static func rope(_ pos: MLXArray, dim: Int, theta: Float) -> MLXArray {
    let scale = MLXArray(0..<dim)[.stride(by: 2)] / Float32(dim)
    let omega = 1.0 / (theta ** scale)
    let batchSize = pos.dim(0)
    let posExpanded = MLX.expandedDimensions(pos, axis: -1)
    let omegaExpanded = MLX.expandedDimensions(omega, axis: 0)
    let out = posExpanded * omegaExpanded
    let cosOut = MLX.cos(out)
    let sinOut = MLX.sin(out)
    let stackedOut = MLX.stacked([cosOut, -sinOut, sinOut, cosOut], axis: -1)
    return MLX.reshaped(stackedOut, [batchSize, -1, dim / 2, 2, 2])
  }
}

public class TextEmbedder: Module {
  @ModuleInfo(key: "linear_1") var linear1: Linear
  @ModuleInfo(key: "linear_2") var linear2: Linear

  init(_ config: MultiModalDiffusionConfiguration) {
    let inputDim = config.pooledProjectionDim
    let hiddenDim: Int = config.numAttentionHeads * config.attentionHeadDim
    self._linear1.wrappedValue = Linear(inputDim, hiddenDim)
    self._linear2.wrappedValue = Linear(hiddenDim, hiddenDim)
  }

  func callAsFunction(_ caption: MLXArray) -> MLXArray {
    var hiddenStates = linear1(caption)
    hiddenStates = MLXNN.silu(hiddenStates)
    hiddenStates = linear2(hiddenStates)
    return hiddenStates
  }
}
public class GuidanceEmbedder: Module {
  @ModuleInfo(key: "linear_1") var linear1: Linear
  @ModuleInfo(key: "linear_2") var linear2: Linear

  init(_ config: MultiModalDiffusionConfiguration) {
    let hiddenDim: Int = config.numAttentionHeads * config.attentionHeadDim
    let inChannels: Int = 256
    self._linear1.wrappedValue = Linear(inChannels, hiddenDim)
    self._linear2.wrappedValue = Linear(hiddenDim, hiddenDim)
  }

  func callAsFunction(_ sample: MLXArray) -> MLXArray {
    var output = linear1(sample)
    output = MLXNN.silu(output)
    output = linear2(output)
    return output
  }
}
public class TimestepEmbedder: Module {
  @ModuleInfo(key: "linear_1") var linear1: Linear
  @ModuleInfo(key: "linear_2") var linear2: Linear

  init(_ config: MultiModalDiffusionConfiguration) {
    let hiddenDim: Int = config.numAttentionHeads * config.attentionHeadDim
    let inChannels: Int = 256
    self._linear1.wrappedValue = Linear(inChannels, hiddenDim)
    self._linear2.wrappedValue = Linear(hiddenDim, hiddenDim)
  }

  func callAsFunction(_ sample: MLXArray) -> MLXArray {
    var output = linear1(sample)
    output = MLXNN.silu(output)
    output = linear2(output)
    return output
  }
}

public class TimeTextEmbed: Module {
  @ModuleInfo(key: "text_embedder") var textEmbedder: TextEmbedder
  @ModuleInfo(key: "guidance_embedder") var guidanceEmbedder: GuidanceEmbedder?
  @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: TimestepEmbedder

  init(_ config: MultiModalDiffusionConfiguration) {
    self._textEmbedder.wrappedValue = TextEmbedder(config)
    self._guidanceEmbedder.wrappedValue =
      config.guidanceEmbeds ? GuidanceEmbedder(config) : nil
    self._timestepEmbedder.wrappedValue = TimestepEmbedder(config)
  }

  func callAsFunction(timeStep: MLXArray, pooledProjection: MLXArray, guidance: MLXArray)
    -> MLXArray
  {
    let timeStepsProj = TimeTextEmbed.timeProj(timeStep)
    var timeStepsEmb = timestepEmbedder(timeStepsProj)
    if let guidanceEmbedder = guidanceEmbedder {
      timeStepsEmb += guidanceEmbedder(TimeTextEmbed.timeProj(guidance))
    }
    let pooledProjections = textEmbedder(pooledProjection)
    let conditioning = timeStepsEmb + pooledProjections
    return conditioning
  }

  static func timeProj(_ timeSteps: MLXArray) -> MLXArray {
    let maxPeriod: Float = 10000
    let halfDim = 128
    let exponent =
      -log(maxPeriod) * MLXArray(0..<halfDim).asType(.float32)
      / Float32(halfDim)
    var emb = MLX.exp(exponent)
      emb = timeSteps[.ellipsis, .newAxis].asType(.float32) * emb[.newAxis]
    emb = MLX.concatenated([MLX.sin(emb), MLX.cos(emb)], axis: -1)
    emb = MLX.concatenated([emb[.ellipsis, halfDim...], emb[.ellipsis, ..<halfDim]], axis: -1)
    return emb
  }
}
public class AdaLayerNormZero: Module {
  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "norm") var norm: LayerNorm

  init(_ modelConfig: MultiModalDiffusionConfiguration) {
    let hiddenDim: Int = modelConfig.numAttentionHeads * modelConfig.attentionHeadDim
    self._linear.wrappedValue = Linear(hiddenDim, hiddenDim * 6)
    self._norm.wrappedValue = LayerNorm(
      dimensions: hiddenDim, eps: modelConfig.layerNormEps, affine: false)
  }

  func callAsFunction(_ x: MLXArray, _ textEmbeddings: MLXArray) -> (
    MLXArray, MLXArray, MLXArray, MLXArray, MLXArray
  ) {
    let textEmbeddings = linear(MLXNN.silu(textEmbeddings))
    let shiftMsa = textEmbeddings[0..., 0..<3072]
    let scaleMsa = textEmbeddings[0..., 3072..<6144]
    let gateMsa = textEmbeddings[0..., 6144..<9216]
    let shiftMlp = textEmbeddings[0..., 9216..<12288]
    let scaleMlp = textEmbeddings[0..., 12288..<15360]
    let gateMlp = textEmbeddings[0..., 15360..<18432]

    let normalizedX = norm(x)
    let output =
      normalizedX * (1 + scaleMsa[0..., .newAxis])
      + shiftMsa[0..., .newAxis]
    return (output, gateMsa, shiftMlp, scaleMlp, gateMlp)
  }
}

public class ProjLinear: Module, UnaryLayer {
  @ModuleInfo(key: "proj") var linear: Linear

  init(_ hiddenDim: Int) {
    self._linear.wrappedValue = Linear(hiddenDim, 2 * hiddenDim)
  }
  public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    return linear(hiddenStates)
  }
}

public class FeedForward: Module {
  @ModuleInfo(key: "linear1") var linear1: Linear
  @ModuleInfo(key: "linear2") var linear2: Linear
  let activation: (MLXArray) -> MLXArray

  init(_ config: MultiModalDiffusionConfiguration, activation: @escaping (MLXArray) -> MLXArray) {
    let hiddenDim: Int = config.numAttentionHeads * config.attentionHeadDim
    self._linear1.wrappedValue = Linear(hiddenDim, 2 * hiddenDim)
    self.activation = activation
    self._linear2.wrappedValue = Linear(2 * hiddenDim, hiddenDim)
  }

  func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    let x = linear1(hiddenStates)
    let y = activation(x)
    let z = linear2(y)
    return z
  }
}

public class JointAttention: Module {
  let config: MultiModalDiffusionConfiguration

  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out") var toOut: [Linear]
  @ModuleInfo(key: "add_q_proj") var addQProj: Linear
  @ModuleInfo(key: "add_k_proj") var addKProj: Linear
  @ModuleInfo(key: "add_v_proj") var addVProj: Linear
  @ModuleInfo(key: "to_add_out") var toAddOut: Linear
  @ModuleInfo(key: "norm_q") var normQ: RMSNorm
  @ModuleInfo(key: "norm_k") var normK: RMSNorm
  @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
  @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm

  init(_ config: MultiModalDiffusionConfiguration) {
    self.config = config

    let hiddenSize = config.numAttentionHeads * config.attentionHeadDim
    self._toQ.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._toK.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._toV.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._toOut.wrappedValue = [Linear(hiddenSize, hiddenSize)]
    self._addQProj.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._addKProj.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._addVProj.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._toAddOut.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._normQ.wrappedValue = RMSNorm(dimensions: config.attentionHeadDim)
    self._normK.wrappedValue = RMSNorm(dimensions: config.attentionHeadDim)
    self._normAddedQ.wrappedValue = RMSNorm(dimensions: config.attentionHeadDim)
    self._normAddedK.wrappedValue = RMSNorm(dimensions: config.attentionHeadDim)
  }

  func callAsFunction(
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    imageRotaryEmb: MLXArray
  ) -> (MLXArray, MLXArray) {

    var query = toQ(hiddenStates)
    var key = toK(hiddenStates)
    var value = toV(hiddenStates)

    query = query.reshaped(1, -1, config.numAttentionHeads, config.attentionHeadDim).transposed(
      0, 2, 1, 3)
    key = key.reshaped(1, -1, config.numAttentionHeads, config.attentionHeadDim).transposed(
      0, 2, 1, 3)
    value = value.reshaped(1, -1, config.numAttentionHeads, config.attentionHeadDim).transposed(
      0, 2, 1, 3)

    query = normQ(query)
    key = normK(key)

    var encoderHiddenStatesQueryProj = addQProj(encoderHiddenStates)
    var encoderHiddenStatesKeyProj = addKProj(encoderHiddenStates)
    var encoderHiddenStatesValueProj = addVProj(encoderHiddenStates)

    encoderHiddenStatesQueryProj = encoderHiddenStatesQueryProj.reshaped(
      1, -1, config.numAttentionHeads, config.attentionHeadDim
    ).transposed(0, 2, 1, 3)
    encoderHiddenStatesKeyProj = encoderHiddenStatesKeyProj.reshaped(
      1, -1, config.numAttentionHeads, config.attentionHeadDim
    ).transposed(0, 2, 1, 3)
    encoderHiddenStatesValueProj = encoderHiddenStatesValueProj.reshaped(
      1, -1, config.numAttentionHeads, config.attentionHeadDim
    ).transposed(0, 2, 1, 3)

    encoderHiddenStatesQueryProj = normAddedQ(encoderHiddenStatesQueryProj)
    encoderHiddenStatesKeyProj = normAddedK(encoderHiddenStatesKeyProj)

    query = MLX.concatenated([encoderHiddenStatesQueryProj, query], axis: 2)
    key = MLX.concatenated([encoderHiddenStatesKeyProj, key], axis: 2)
    value = MLX.concatenated([encoderHiddenStatesValueProj, value], axis: 2)
    (query, key) = JointAttention.applyRope(query, key, freqsCis: imageRotaryEmb)

    var hiddenStates = MLXFast.scaledDotProductAttention(queries: query, keys: key, values: value, scale: 1 / sqrt(Float(query.dim(-1))),mask: nil)

    hiddenStates = hiddenStates.transposed(0, 2, 1, 3)
    hiddenStates = hiddenStates.reshaped(1, -1, config.numAttentionHeads * config.attentionHeadDim)

    let splitIndex = encoderHiddenStates.dim(1)
    let encoderOutput = hiddenStates[0..., 0..<splitIndex, 0...]
    hiddenStates = hiddenStates[0..., splitIndex..., 0...]

    hiddenStates = toOut[0](hiddenStates)
    let encoderHiddenStatesOutput = toAddOut(encoderOutput)
    return (hiddenStates, encoderHiddenStatesOutput)
  }

  static func applyRope(_ xq: MLXArray, _ xk: MLXArray, freqsCis: MLXArray) -> (MLXArray, MLXArray)
  {
    let xq_ = xq.asType(.float32).reshaped(xq.shape.dropLast() + [-1, 1, 2])
    let xk_ = xk.asType(.float32).reshaped(xk.shape.dropLast() + [-1, 1, 2])
    let xqOut: MLXArray =
      freqsCis[.ellipsis, 0] * xq_[.ellipsis, 0] + freqsCis[.ellipsis, 1] * xq_[.ellipsis, 1]
    let xkOut =
      freqsCis[.ellipsis, 0] * xk_[.ellipsis, 0] + freqsCis[.ellipsis, 1] * xk_[.ellipsis, 1]

    return (xqOut.reshaped(xq.shape).asType(.float32), xkOut.reshaped(xk.shape).asType(.float32))
  }
}

public class JointTransformerBlock: Module {
  @ModuleInfo(key: "norm1") var norm1: AdaLayerNormZero
  @ModuleInfo(key: "norm2") var norm2: LayerNorm
  @ModuleInfo(key: "ff") var ff: FeedForward
  @ModuleInfo(key: "attn") var attn: JointAttention
  @ModuleInfo(key: "norm1_context") var norm1Context: AdaLayerNormZero
  @ModuleInfo(key: "ff_context") var ffContext: FeedForward
  @ModuleInfo(key: "norm2_context") var norm2Context: LayerNorm

  init(_ config: MultiModalDiffusionConfiguration) {
    let hiddenDim: Int = config.numAttentionHeads * config.attentionHeadDim
    self._norm1.wrappedValue = AdaLayerNormZero(config)
    self._norm2.wrappedValue = LayerNorm(
      dimensions: hiddenDim, eps: config.layerNormEps, affine: false)
    self._ff.wrappedValue = FeedForward(config, activation: geluApproximate)
    self._attn.wrappedValue = JointAttention(config)
    self._norm1Context.wrappedValue = AdaLayerNormZero(config)
    self._ffContext.wrappedValue = FeedForward(config, activation: geluApproximate)
    self._norm2Context.wrappedValue = LayerNorm(
      dimensions: hiddenDim, eps: config.layerNormEps, affine: false)
  }

  func callAsFunction(
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    textEmbeddings: MLXArray,
    rotaryEmbeddings: MLXArray
  ) -> (MLXArray, MLXArray) {
    let (normHiddenStates, gateMsa, shiftMlp, scaleMlp, gateMlp) = norm1(
        hiddenStates, textEmbeddings)

    let (normEncoderHiddenStates, cGateMsa, cShiftMlp, cScaleMlp, cGateMlp) = norm1Context(
        encoderHiddenStates,
        textEmbeddings
    )
    
    let (attnOutput, contextAttnOutput) = attn(
        hiddenStates: normHiddenStates,
        encoderHiddenStates: normEncoderHiddenStates,
        imageRotaryEmb: rotaryEmbeddings
    )

    var newHiddenStates = hiddenStates + MLX.expandedDimensions(gateMsa, axis: 1) * attnOutput
    var normNewHiddenStates = norm2(newHiddenStates)

    normNewHiddenStates =
        normNewHiddenStates * (1 + MLX.expandedDimensions(scaleMlp, axis: 1))
        + MLX.expandedDimensions(shiftMlp, axis: 1)

    let ffOutput = ff(normNewHiddenStates)

    newHiddenStates = newHiddenStates + MLX.expandedDimensions(gateMlp, axis: 1) * ffOutput

    var newEncoderHiddenStates =
        encoderHiddenStates + MLX.expandedDimensions(cGateMsa, axis: 1) * contextAttnOutput

    var normNewEncoderHiddenStates = norm2Context(newEncoderHiddenStates)

    normNewEncoderHiddenStates =
        normNewEncoderHiddenStates * (1 + MLX.expandedDimensions(cScaleMlp, axis: 1))
        + MLX.expandedDimensions(cShiftMlp, axis: 1)


    let contextFfOutput = ffContext(normNewEncoderHiddenStates)


    newEncoderHiddenStates =
        newEncoderHiddenStates + MLX.expandedDimensions(cGateMlp, axis: 1) * contextFfOutput

    return (newEncoderHiddenStates, newHiddenStates)
  }
}
public class AdaLayerNormContinuous: Module {
  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "norm") var norm: LayerNorm
  let embeddingDim: Int

  init(_ config: MultiModalDiffusionConfiguration, embeddingDim: Int, conditioningEmbeddingDim: Int)
  {
    self.embeddingDim = embeddingDim
    self._linear.wrappedValue = Linear(conditioningEmbeddingDim, embeddingDim * 2)
    self._norm.wrappedValue = LayerNorm(
      dimensions: embeddingDim, eps: config.layerNormEps, affine: false)
  }

  func callAsFunction(_ x: MLXArray, _ textEmbeddings: MLXArray) -> MLXArray {
    let textEmbeddings = linear(MLXNN.silu(textEmbeddings))
    let chunkSize = embeddingDim

    let scale = textEmbeddings[0..., 0 * chunkSize..<1 * chunkSize]
    let shift = textEmbeddings[0..., 1 * chunkSize..<2 * chunkSize]

    let normalizedX = norm(x)
    let output =
      normalizedX * MLX.expandedDimensions((1 + scale), axis: 1)
      + MLX.expandedDimensions(shift, axis: 1)

    return output
  }
}

public class AdaLayerNormZeroSingle: Module {
  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "norm") var norm: LayerNorm
  let hiddenDim: Int

  init(_ config: MultiModalDiffusionConfiguration) {
    self.hiddenDim = config.numAttentionHeads * config.attentionHeadDim
    self._linear.wrappedValue = Linear(self.hiddenDim, 3 * self.hiddenDim)
    self._norm.wrappedValue = LayerNorm(
      dimensions: self.hiddenDim, eps: config.layerNormEps, affine: false)
  }

  func callAsFunction(x: MLXArray, textEmbeddings: MLXArray) -> (MLXArray, MLXArray) {
    let textEmbeddings = linear(MLXNN.silu(textEmbeddings))
    let chunkSize = self.hiddenDim

    let shiftMsa = textEmbeddings[0..., 0 * chunkSize..<1 * chunkSize]
    let scaleMsa = textEmbeddings[0..., 1 * chunkSize..<2 * chunkSize]
    let gateMsa = textEmbeddings[0..., 2 * chunkSize..<3 * chunkSize]

    let normalizedX = norm(x)
    let output =
      normalizedX * MLX.expandedDimensions((1+scaleMsa), axis: 1)
      + MLX.expandedDimensions(shiftMsa, axis: 1)
    return (output, gateMsa)
  }
}

public class SingleTransformerBlock: Module {
  @ModuleInfo(key: "norm") var norm: AdaLayerNormZeroSingle
  @ModuleInfo(key: "proj_mlp") var projMlp: Linear
  @ModuleInfo(key: "attn") var attn: SingleBlockAttention
  @ModuleInfo(key: "proj_out") var projOut: Linear

  init(_ config: MultiModalDiffusionConfiguration) {
    let hiddenDim: Int = config.numAttentionHeads * config.attentionHeadDim
    self._norm.wrappedValue = AdaLayerNormZeroSingle(config)
    self._projMlp.wrappedValue = Linear(hiddenDim, 4 * hiddenDim)
    self._attn.wrappedValue = SingleBlockAttention(config)
    self._projOut.wrappedValue = Linear(hiddenDim + 4 * hiddenDim, hiddenDim)
  }

  func callAsFunction(
    hiddenStates: MLXArray,
    textEmbeddings: MLXArray,
    rotaryEmbeddings: MLXArray
  ) -> MLXArray {
    let residual = hiddenStates
    let (normHiddenStates, gate) = norm(x: hiddenStates, textEmbeddings: textEmbeddings)
    let mlpHiddenStates = MLXNN.geluFastApproximate(projMlp(normHiddenStates))
    let attnOutput = attn(
      hiddenStates: normHiddenStates,
      imageRotaryEmb: rotaryEmbeddings
    )
    var hiddenStates = MLX.concatenated([attnOutput, mlpHiddenStates], axis: 2)
    let gateExpanded = MLX.expandedDimensions(gate, axis: 1)
    hiddenStates = gateExpanded * projOut(hiddenStates)
    hiddenStates = residual + hiddenStates
    return hiddenStates
  }
}

public class SingleBlockAttention: Module {
  let config: MultiModalDiffusionConfiguration

  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "norm_q") var normQ: RMSNorm
  @ModuleInfo(key: "norm_k") var normK: RMSNorm

  init(_ config: MultiModalDiffusionConfiguration) {
    self.config = config
    let hiddenDim = config.numAttentionHeads * config.attentionHeadDim
    self._toQ.wrappedValue = Linear(hiddenDim, hiddenDim)
    self._toK.wrappedValue = Linear(hiddenDim, hiddenDim)
    self._toV.wrappedValue = Linear(hiddenDim, hiddenDim)
    self._normQ.wrappedValue = RMSNorm(dimensions: config.attentionHeadDim)
    self._normK.wrappedValue = RMSNorm(dimensions: config.attentionHeadDim)
  }

  func callAsFunction(
    hiddenStates: MLXArray,
    imageRotaryEmb: MLXArray
  ) -> MLXArray {
    var query = toQ(hiddenStates)
    var key = toK(hiddenStates)
    var value = toV(hiddenStates)

    query = query.reshaped(1, -1, config.numAttentionHeads, config.attentionHeadDim).transposed(
      0, 2, 1, 3)
    key = key.reshaped(1, -1, config.numAttentionHeads, config.attentionHeadDim).transposed(
      0, 2, 1, 3)
    value = value.reshaped(1, -1, config.numAttentionHeads, config.attentionHeadDim).transposed(
      0, 2, 1, 3)

    query = normQ(query)
    key = normK(key)

    (query, key) = SingleBlockAttention.applyRope(query, key, freqsCis: imageRotaryEmb)

    let scale = pow(Float(config.attentionHeadDim), -0.5)
    var hiddenStates = MLXFast.scaledDotProductAttention(
      queries: query,
      keys: key,
      values: value,
      scale: scale,
      mask: nil
    )
    hiddenStates = hiddenStates.transposed(0, 2, 1, 3)
    hiddenStates = hiddenStates.reshaped(1, -1, config.numAttentionHeads * config.attentionHeadDim)

    return hiddenStates
  }

  static func applyRope(_ xq: MLXArray, _ xk: MLXArray, freqsCis: MLXArray) -> (MLXArray, MLXArray)
  {
    let xq_ = xq.asType(.float32).reshaped(xq.shape.dropLast() + [-1, 1, 2])
    let xk_ = xk.asType(.float32).reshaped(xk.shape.dropLast() + [-1, 1, 2])

    let xqOut =
      freqsCis[.ellipsis, 0] * xq_[.ellipsis, 0] + freqsCis[.ellipsis, 1] * xq_[.ellipsis, 1]
    let xkOut =
      freqsCis[.ellipsis, 0] * xk_[.ellipsis, 0] + freqsCis[.ellipsis, 1] * xk_[.ellipsis, 1]

    return (xqOut.reshaped(xq.shape).asType(.float32), xkOut.reshaped(xk.shape).asType(.float32))
  }
}

public class MultiModalDiffusionTransformer: Module {
  @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
  @ModuleInfo(key: "pos_embed") var posEmbed: EmbedND
  @ModuleInfo(key: "time_text_embed") var timeTextEmbed: TimeTextEmbed
  @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
  @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [JointTransformerBlock]
  @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks:
    [SingleTransformerBlock]
  @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
  @ModuleInfo(key: "proj_out") var projOut: Linear

  public init(_ config: MultiModalDiffusionConfiguration) {
    let innerDim = config.numAttentionHeads * config.attentionHeadDim
    let outChannels = config.inChannels

    self._xEmbedder.wrappedValue = Linear(outChannels, innerDim)
    self._posEmbed.wrappedValue = EmbedND(config)
    self._timeTextEmbed.wrappedValue = TimeTextEmbed(config)
    self._contextEmbedder.wrappedValue = Linear(config.jointAttentionDim, innerDim)
    self._transformerBlocks.wrappedValue = (0..<config.numLayers).map { _ in
      JointTransformerBlock(config)
    }
    self._singleTransformerBlocks.wrappedValue = (0..<config.numSingleLayers).map { _ in
      SingleTransformerBlock(config)
    }
    self._normOut.wrappedValue = AdaLayerNormContinuous(
      config, embeddingDim: innerDim, conditioningEmbeddingDim: innerDim)
    self._projOut.wrappedValue = Linear(innerDim, outChannels)
  }

    public func callAsFunction(
    t: Int,
    promptEmbeds: MLXArray,
    pooledPromptEmbeds: MLXArray,
    hiddenStates: MLXArray,
    evaluateParameters: EvaluateParameters,
    imgIds: MLXArray? = nil,
    controlnetBlockSamples: [MLXArray]? = nil,
    controlnetSingleBlockSamples: [MLXArray]? = nil
  ) -> MLXArray {
    let xType = hiddenStates.dtype
    let timeStep = evaluateParameters.sigmas[t] * Float(evaluateParameters.numTrainSteps)
    let timeStepArray = MLX.broadcast(timeStep, to: [1]).asType(xType)
    var hiddenStates = xEmbedder(hiddenStates)
    let guidance = MLXArray([evaluateParameters.guidance * Float(evaluateParameters.numTrainSteps)]).asType(xType)

    let textEmbeddings = timeTextEmbed(
      timeStep: timeStepArray, pooledProjection: pooledPromptEmbeds, guidance: guidance)
    var encoderHiddenStates = contextEmbedder(promptEmbeds)
    let txtIds = MultiModalDiffusionTransformer.prepareTextIds(seqLen: promptEmbeds.dim(1))
      
    var imageIds = MultiModalDiffusionTransformer.prepareLatentImageIds(
      height: evaluateParameters.height, width: evaluateParameters.width)
    
    if let kontextImgIds = imgIds {
      imageIds = MLX.concatenated([imageIds, kontextImgIds], axis: 1)
    }
    
    let ids = MLX.concatenated([txtIds, imageIds], axis: 1)
    let imageRotaryEmb = posEmbed(ids)

    for (idx, block) in transformerBlocks.enumerated() {
      (encoderHiddenStates, hiddenStates) = block(
        hiddenStates: hiddenStates,
        encoderHiddenStates: encoderHiddenStates,
        textEmbeddings: textEmbeddings,
        rotaryEmbeddings: imageRotaryEmb
      )
      if let controlnetBlockSamples = controlnetBlockSamples, !controlnetBlockSamples.isEmpty {
        let intervalControl = Int(
          ceil(Float(transformerBlocks.count) / Float(controlnetBlockSamples.count)))
        hiddenStates = hiddenStates + controlnetBlockSamples[idx / intervalControl]
      }
    }

    hiddenStates = MLX.concatenated([encoderHiddenStates, hiddenStates], axis: 1)

    for (idx, block) in singleTransformerBlocks.enumerated() {
      hiddenStates = block(
        hiddenStates: hiddenStates,
        textEmbeddings: textEmbeddings,
        rotaryEmbeddings: imageRotaryEmb
      )
      if let controlnetSingleBlockSamples = controlnetSingleBlockSamples,
        !controlnetSingleBlockSamples.isEmpty
      {
        let intervalControl = Int(
          ceil(Float(singleTransformerBlocks.count) / Float(controlnetSingleBlockSamples.count)))
        let encoderShape = encoderHiddenStates.shape[1]
        hiddenStates[0..., encoderShape..., 0...] =
          hiddenStates[0..., encoderShape..., 0...]
          + controlnetSingleBlockSamples[idx / intervalControl]
      }
    }

    hiddenStates = hiddenStates[0..., encoderHiddenStates.shape[1]..., 0...]

    hiddenStates = normOut(hiddenStates, textEmbeddings)
    hiddenStates = projOut(hiddenStates)
    let noise = hiddenStates
    return noise
  }

  public static func prepareLatentImageIds(height: Int, width: Int) -> MLXArray {
    let latentWidth = width / 16
    let latentHeight = height / 16
    var latentImageIds = MLX.zeros([latentHeight, latentWidth, 3])
    latentImageIds[0..., 0..., 1] =
      add(latentImageIds[0..., 0..., 1], MLXArray(0..<latentHeight)[0..., .newAxis])
    latentImageIds[0..., 0..., 2] =
      add(latentImageIds[0..., 0..., 2], MLXArray(0..<latentWidth)[.newAxis, 0...])
    latentImageIds = MLX.repeated(latentImageIds[.newAxis, .ellipsis], count: 1, axis: 0)
    latentImageIds = MLX.reshaped(latentImageIds, [1, latentWidth * latentHeight, 3])
    return latentImageIds
  }

  public static func prepareTextIds(seqLen: Int) -> MLXArray {
    return MLX.zeros([1, seqLen, 3])
  }
}