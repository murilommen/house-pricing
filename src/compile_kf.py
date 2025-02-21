from kfp.compiler import compiler
from src.pipeline import regression_pipeline

pipeline_filename = 'regression_pipeline_v2.json'

compiler.Compiler().compile(
    pipeline_func=regression_pipeline, package_path=pipeline_filename
)

print(f"Compiled pipeline to: {pipeline_filename}")