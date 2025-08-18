python train.py --dataset chexpert --loss focal --max-epochs 1 --batch-size 1 --chexpert-exclude-support-devices --rat-skip-strength 0.3 --rat-width 32 --input-noise-std 0.05

python visualize.py --dataset chexpert --num-samples 4 --ckpt-path C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\outputs\lightning_logs\version_0\checkpoints\best-epoch=00.ckpt --rat-width 32 --rat-skip-strength 0.3

python .\gen_masks.py --sam-checkpoint C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\sam\sam_vit_b_01ec64.pth --output-dir C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\sam --single-file --viz --viz-num-samples 2