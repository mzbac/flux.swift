// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "flux.swift",
  platforms: [.macOS(.v14), .iOS(.v16)],
  products: [
    .library(
      name: "FluxSwift",
      targets: ["FluxSwift"])
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.16.0"),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.13"),
  ],
  targets: [
    .target(
      name: "FluxSwift",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXFast", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXOptimizers", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "Transformers", package: "swift-transformers"),
      ]
    ),
    .testTarget(
      name: "FluxSwiftTests",
      dependencies: ["FluxSwift"],
      path: "Tests"
    ),
  ]
)
