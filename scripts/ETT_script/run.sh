model_name='Transformer ns_Transformer CI_NS_Transformer Informer ns_Informer Autoformer ns_Autoformer'
seqlen='96 144 192 720'
horizons='96 192 336 720'
data_name='ETTh1 ETTh2 ETTm1 ETTm2'

for model in $model_name
do
    for data in $data_name
    do
        for lookback in $seqlen
        do
            for horizon in $horizons
            do
                python -u run.py \
                    --is_training 1 \
                    --root_path ./dataset/ETT-small/ \
                    --data_path $data.csv \
                    --model_id ETT_$lookback_$horizon \
                    --model $model \
                    --data $data \
                    --features M \
                    --seq_len $lookback \
                    --label_len 48 \
                    --pred_len $horizon \
                    --e_layers 2 \
                    --d_layers 1 \
                    --enc_in 7 \
                    --dec_in 7 \
                    --c_out 7 \
                    --gpu 0 \
                    --des 'Exp_h256_l2' \
                    --p_hidden_dims 256 256 \
                    --p_hidden_layers 2 \
                    --itr 1 &
            done
        done
    done
done