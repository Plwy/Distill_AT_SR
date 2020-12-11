# load 参数为训练时的path， 加载上一次训练的log psnr_log.pt
# save_results 自动保存到'experiment/' + args.load 或者'experiment/' + args.save
python main.py --test_only \
                --data_test 'GF' \
                --pre_train 'experiment/BIX2_01mse_lr1e4_41/model/model_latest.pt' \
                --save '267_result' \
                --save_results

