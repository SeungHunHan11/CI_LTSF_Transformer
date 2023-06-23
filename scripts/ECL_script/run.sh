model_name='Transformer ns_Transformer CI_NS_Transformer Informer ns_Informer Autoformer ns_Autoformer'
seqlen='96 144 192 720'
horizons='96 192 336 720'

for model in $model_name
do
    for lookback in $seqlen
    do
        for horizon in $horizons
        do
            python -u run.py \
                --is_training 1 \
                --root_path ./dataset/electricity/ \
                --data_path electricity.csv \
                --model_id ECL_$lookback_$horizon \
                --model $model \
                --data custom \
                --features M \
                --seq_len $lookback \
                --label_len 48 \
                --pred_len $horizon \
                --e_layers 2 \
                --d_layers 1 \
                --factor 3 \
                --enc_in 321 \
                --dec_in 321 \
                --c_out 321 \
                --gpu 0 \
                --des 'Exp_h256_l2' \
                --p_hidden_dims 256 256 \
                --p_hidden_layers 2 \
                --itr 1 &

        done
    done
done