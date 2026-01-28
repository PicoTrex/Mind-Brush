import json
import warnings

from fastmcp import FastMCP


mcp = FastMCP("base-tools")
gee_meta_path = './resource/gee_product_metas.json'

@mcp.resource("gee://products/product_list", description="Get all available GEE's products in list[str] format.")
def get_gee_product_list() -> list[str]:
    '''Fetch all available GEE's products in list[str] format.'''
    with open(gee_meta_path, 'r') as f:
        data = json.load(f)
    return [_['Product_Name'] for _ in data]

@mcp.resource("gee://products/{product_name}", description="Get a specific GEE's product meta in dict format.")
def get_gee_product_meta(product_name: str) -> dict:
    '''Fetch a specific GEE's product meta in dict format.'''
    with open(gee_meta_path, 'r') as f:
        data = json.load(f)
    return [_ for _ in data if _['Product_Name'] == product_name][0]


if __name__ == "__main__":
    mcp.run() 
