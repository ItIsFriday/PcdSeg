## Efficient Point Cloud Segmentation with Geometry-aware Sparse Networks (ECCV2022)
## requirements
- [spconv](https://github.com/traveller59/spconv)
- [torchscatter](https://github.com/rusty1s/pytorch_scatter)

## acknowledgement
- [mmselfsup](https://github.com/open-mmlab/mmselfsup)

## usage
```bash
./tools/dist_train.sh configs/benchmarks/mmsegmentation/semantic_kitti/gasn.py 8 --work_dir work_dirs/logs
```