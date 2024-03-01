```mermaid
classDiagram
    class ovNode
    class NodeExtension
    class TypedNodeExtensionBase {
        <<template< ovCommonOp >>>
    }
    class TypedNodeExtension {
        <<template< ovCommonOp >>>
    }

    class ovCommonOp

    ovNode <|-- ovCommonOp
    NodeExtension <|-- TypedNodeExtensionBase
    TypedNodeExtensionBase <|-- TypedNodeExtension
    ovCommonOp <|-- TypedNodeExtension
    NodeExtension  ..>  TypedNodeExtension : Has pointer
```

---------------------------

```mermaid
classDiagram
    class MemoryDescs {
        + layout
        + precision
        + paddings
    }

    class Model
    class ImplementationsFactory {
        + impls list
    }
    class OpImplementation
    class OptimizationAttributes
    class LayoutOptimizer
    class OpExecutor

    class NodeExtension

    NodeExtension o-- Model : fused ops
    NodeExtension o-- MemoryDescs
    NodeExtension *-- ImplementationsFactory
    NodeExtension o-- OptimizationAttributes
    NodeExtension o-- LayoutOptimizer
    NodeExtension o-- "best impl" OpImplementation
    NodeExtension o-- "best executor" OpExecutor

```
