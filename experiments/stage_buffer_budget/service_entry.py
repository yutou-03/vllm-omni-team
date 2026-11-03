"""Use the frozen experiment source ahead of container editable installations."""
import os,sys
from pathlib import Path
framework=Path(os.environ['STAGE_STUDY_FRAMEWORK']).resolve()
sys.path.insert(0,str(framework))
import vllm_omni
assert Path(vllm_omni.__file__).resolve().is_relative_to(framework)
from vllm.entrypoints.cli.main import main
if __name__=='__main__':main()
