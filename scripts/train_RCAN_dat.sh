"""RCAN_dat模型"""
# # DIV2K数据集上的训练
# python main.py --dir_data '/media/zsl/data/zsl_datasets/RCAN/traindata/DIV2K/bicubic' \
#                 --data_train 'DIV2K' \
#                 --data_test 'DIV2K' \
#                 --n_train 30 \
#                 --offset_val 30 \
#                 --model 'RCAN_AT' \
#                 --teacher_ckpts 'ckpts/teacher_ckpts/CRAFT_model/craft_mlt_25k.pth' \
#                 --loss '1*MSE' \
#                 --feature_distilation_type '1*SD+1*AD' \
#                 --save 'RCAN_dat_train'

# 高法数据集上的训练
python main.py --dir_data '/media/zsl/data/zsl_datasets/RCAN/traindata/DIV2K/bicubic' \
                --data_train 'DIV2K' \
                --data_test 'DIV2K' \
                --n_train 30 \
                --offset_val 30 \
                --model 'RCAN_AT' \ 
                --teacher_ckpts 'ckpts/teacher_ckpts/CRAFT_model/craft_mlt_25k.pth' \
                --loss '1*MSE' \
                --feature_distilation_type '1*SD+1*AD' \
                --save 'RCAN_dat_train'




