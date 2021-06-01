echo Using GPU: [$1]
echo Attacker Name: [$2]

CUDA_VISIBLE_DEVICES=$1 python3 run_attacks.py --attacks attacker --output tmp --name $2 --models "imagenet-fast_at","imagenet-free_at"