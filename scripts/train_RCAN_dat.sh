# # # DIV2K数据集上的训练
# python main.py --dir_data '/media/zsl/data/zsl_datasets/RCAN/traindata/DIV2K/bicubic/DIV2K_test' \
#                 --data_train 'DIV2K' \
#                 --data_test 'DIV2K' \
#                 --n_train 30 \
#                 --offset_val 30 \
#                 --model 'RCAN_AT' \
#                 --teacher_ckpts 'ckpts/teacher_ckpts/CRAFT_model/craft_mlt_25k.pth' \
#                 --loss '0.1*L1' \
#                 --feature_distilation_type '1*SD+1*AD' \
#                 --save 'DIV2k_train' \
#                 --pre_train experiment/pre_trained/RCAN_BIX2.pt \
#                 --scale 4 \
#                 --save_results


                
# # 小数据集测试用
# python main.py  --dir_data '/media/zsl/data/zsl_datasets/GF/sm_data_set' \
#                 --pre_train 'experiment/pre_trained/RCAN_BIX2.pt' \
#                 --model 'RCAN_AT' \
#                 --teacher_ckpts 'ckpts/teacher_ckpts/CRAFT_model/craft_mlt_25k.pth' \
#                 --loss '0.1*MSE' \
#                 --feature_distilation_type '1*SD+1*AD' \
#                 --batch_size 8 \
#                 --scale 2 \
#                 --save 'best_test' \
#                 --save_results \
#                 --reset
            

# 高法数据集上的训练
python main.py  --dir_data '/media/zsl/data/zsl_datasets/GF/data_set' \
                --pre_train 'experiment/pre_trained/RCAN_BIX2.pt' \
                --model 'RCAN_AT' \
                --teacher_ckpts 'ckpts/teacher_ckpts/CRAFT_model/craft_mlt_25k.pth' \
                --loss '1*L1' \
                --feature_distilation_type '1*SD+1*AD' \
                --batch_size 8 \
                --scale 2 \
                --save_step 5 \
                --save_results \
                --save 'RCAN_DAT_BIX2_TRAIN_1L1_lre6' \
                --lr 1e-6