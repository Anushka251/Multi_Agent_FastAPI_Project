
from pathlib import Path

from app.agents import graph

OUTPUT_DIR = Path("screenshots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Print the actual graph structure
print("FULL LANGGRAPH FLOW")
print("===================")
print(graph.get_graph().draw_mermaid())

# Save the Mermaid graph source for documentation
mermaid_code = graph.get_graph().draw_mermaid()
(OUTPUT_DIR / "langgraph_flow.mmd").write_text(
    mermaid_code,
    encoding="utf-8",
)

# Save a PNG visualization if Graphviz dependencies are available
try:
    image_bytes = graph.get_graph().draw_mermaid_png()
    image_path = OUTPUT_DIR / "langgraph_flow.png"
    image_path.write_bytes(image_bytes)
    print(f"\nGraph image saved to: {image_path.resolve()}")
except Exception as error:
    print("\nPNG generation was unavailable:", error)
    print("The Mermaid graph source was saved instead.")
