import os
from pathlib import Path
import torch, gc

# ── 캐시 경로 고정 ───────────────────────────────
os.environ['HF_HOME'] = '/source/changmin/Hunyuan3D-2/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/source/changmin/Hunyuan3D-2/hf_cache'

# ── Hunyuan3D-2 파이프라인 로드 (1회) ───────────
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
paint_pipe = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')

# ── 입·출력 폴더 설정 ───────────────────────────
image_dir   = Path('rm_fish')            # 이미지들이 들어 있는 폴더
output_dir  = Path('/data/changmin/outputs')           # 결과를 저장할 폴더
output_dir.mkdir(parents=True, exist_ok=True)

# ── 지원 확장자 목록 ────────────────────────────
valid_exts = {'.png', '.jpg', '.jpeg', '.webp'}

# ── 반복 처리 ───────────────────────────────────
for img_path in sorted(image_dir.iterdir()):
    if img_path.suffix.lower() not in valid_exts:
        continue

    print(f'▶ {img_path.name} 처리 중 …')

    # 1) 메쉬 생성
    mesh = shape_pipe(image=str(img_path))[0]

    # 2) 텍스처 페인팅
    mesh = paint_pipe(mesh, image=str(img_path))

    # 3) GLB 저장
    out_path = output_dir / f'{img_path.stem}.glb'
    mesh.export(out_path)
    print(f'   ↳ 저장 완료: {out_path}')

    # 4) 메모리 정리
    del mesh
    torch.cuda.empty_cache()
    gc.collect()

print('✅ 모든 이미지 처리 완료!')