CUDA_VISIBLE_DEVICES=7 python eval.py -d webqsp -p /egr/research-dselab/guokai1/workspace/SubgraphRAG/retrieve/webqsp_Nov19-23:31:06/retrieval_result.pth


CUDA_VISIBLE_DEVICES=4 nohup python main.py -d cwq --prompt_mode scored_100 -m gpt-4o-mini > cwq_confident.log 2>&1



CUDA_VISIBLE_DEVICES=0 ./scripts/rog-reasoning.sh FI 0.2 