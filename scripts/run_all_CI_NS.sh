data_set='ETTh1 ETTh2 ETTm1 ETTm2 traffic electricity'
model_name='CI_NS_Transformer'
seqlen='96 144 192 720'
horizons='96 192 336 720'
loss='l1 l2'

for data in $data_set
do
    for model in $model_name
    do
        for lookback in $seqlen
        do
            for horizon in $horizons
            do
                for lss in $loss
                do
                    if [ "$data" = "electricity" ]; then
                        path=electricity
                        enc_in=321
                        dec_in=321
                        c_out=321
                    elif [ "$data" = "traffic" ]; then
                        path=traffic
                        enc_in=862
                        dec_in=862
                        c_out=862
                    elif [ "$data" = "weather" ]; then
                        path=weather
                        enc_in=21
                        dec_in=21
                        c_out=21
                    elif [ "$data" = "exchange_rate" ]; then
                        path=exchange_rate
                        enc_in=8
                        dec_in=8
                        c_out=8
                    elif [ "$data" = "national_illness" ]; then
                        path=illness
                        enc_in=7
                        dec_in=7
                        c_out=7
                    else
                        path=ETT-small
                        enc_in=7
                        dec_in=7
                        c_out=7
                    fi

                    model_id="$data-$lookback-$horizon"
                    echo $model_id

                    if [ "$lss" = "l1" ]; then
                        python -u run.py \
                            --is_training 1 \
                            --root_path ./dataset/$path/ \
                            --data_path $data.csv \
                            --model_id $model_id \
                            --model $model \
                            --data custom \
                            --features M \
                            --seq_len $lookback \
                            --batch_size 16 \
                            --label_len 48 \
                            --pred_len $horizon \
                            --e_layers 2 \
                            --d_layers 1 \
                            --factor 3 \
                            --enc_in $enc_in \
                            --dec_in $dec_in \
                            --c_out $c_out \
                            --gpu 0 \
                            --devices 0,1 \
                            --des 'Exp_h256_l2' \
                            --p_hidden_dims 256 256 \
                            --p_hidden_layers 2 \
                            --itr 1 \
                            --use_multi_gpu \
                            --result_name CI_NS_Result
                    else
                        python -u run.py \
                            --is_training 1 \
                            --root_path ./dataset/$path/ \
                            --data_path $data.csv \
                            --model_id $model_id \
                            --model $model \
                            --data custom \
                            --features M \
                            --seq_len $lookback \
                            --batch_size 16 \
                            --label_len 48 \
                            --pred_len $horizon \
                            --e_layers 2 \
                            --d_layers 1 \
                            --factor 3 \
                            --enc_in $enc_in \
                            --dec_in $dec_in \
                            --c_out $c_out \
                            --gpu 0 \
                            --devices 0,1 \
                            --des 'Exp_h256_l2' \
                            --p_hidden_dims 256 256 \
                            --p_hidden_layers 2 \
                            --itr 1 \
                            --use_multi_gpu \
                            --use_l2 \
                            --result_name CI_NS_Result
                    fi
                done
            done
        done
    done
done