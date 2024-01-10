import subprocess

def run_script(script, args):
    """
    Run a script with given arguments using subprocess.

    :param script: Path to the script (e.g., 'train.py')
    :param args: List of arguments (e.g., ['--data', 'data/flickr8k', '--epochs', '4'])
    """
    command = ["python", script] + args
    subprocess.run(command)

def main():
    # Define the arguments for each script call
    plain_att = [
        '--data', 'data/flickr8k',
        '--epochs', '8',
        '--frac', '1',
        '--log-interval', '50',
        '--attention', 
        '--tf',
        '--ado',
        # '--bert'
    ]
   
    plain_noatt = [
        '--data', 'data/flickr8k',
        '--epochs', '8',
        '--frac', '1',
        '--log-interval', '50',
        # '--attention', 
        '--tf',
        '--ado',
        # '--bert'
    ]

    bert_att = [
        '--data', 'data/flickr8k',
        '--epochs', '8',
        '--frac', '1',
        '--log-interval', '50',
        '--attention', 
        '--tf',
        '--ado',
        '--bert'
    ]

    bert_noatt = [
        '--data', 'data/flickr8k',
        '--epochs', '8',
        '--frac', '1',
        '--log-interval', '50',
        # '--attention', 
        '--tf',
        '--ado',
        '--bert'
    ]

    plain_exp_bs = [
        '--data', 'data/flickr8k',
        '--epochs', '35',
        '--frac', '1',
        '--log-interval', '50',
        '--attention', 
        '--tf',
        '--ado',
        '--batch-size', '128',
        # '--bert'
    ]

    plain_exp_lr = [
        '--data', 'data/flickr8k',
        '--epochs', '8',
        '--frac', '1',
        '--log-interval', '50',
        '--attention', 
        '--tf',
        '--ado',
        '--batch-size', '64',
        '--lr', '0.00001',
        # '--bert'
    ]

    plain_exp_lr2 = [
        '--data', 'data/flickr8k',
        '--epochs', '8',
        '--frac', '1',
        '--log-interval', '50',
        '--attention', 
        '--tf',
        '--ado',
        '--batch-size', '64',
        '--lr', '0.001',
        # '--bert'
    ]

    plain_exp_lr2_bert = [
        '--data', 'data/flickr8k',
        '--epochs', '16',
        '--frac', '1',
        '--log-interval', '50',
        '--attention', 
        '--tf',
        '--ado',
        '--batch-size', '64',
        '--lr', '0.001',
        '--bert'
    ]

    plain_exp_lr2_longer = [
        '--data', 'data/flickr8k',
        '--epochs', '12',
        '--frac', '1',
        '--log-interval', '50',
        '--attention', 
        '--tf',
        '--ado',
        '--batch-size', '64',
        '--lr', '0.001',
        '--model', 'model/cache_wandb/tuzy19bt/model/model_vgg19_8.pth'
        # '--bert'
    ]

    plain_noatt_attention_layers_removed = [
        '--data', 'data/flickr8k',
        '--epochs', '8',
        '--frac', '1',
        '--log-interval', '50',
        # '--attention', 
        '--tf',
        '--ado',
        '--batch-size', '64',
        '--lr', '0.0001',
        # '--bert'
    ]

    # Calling the train.py script with the specified arguments
    # print("Running plain_att")
    # run_script('train.py', plain_att)
    # print("Running plain_noatt")
    # run_script('train.py', plain_noatt)
    # print("Running bert_att")
    # run_script('train.py', bert_att)
    # print("Running bert_noatt")
    # run_script('train.py', bert_noatt)
    # print("Running plain_exp_bs")
    # run_script('train.py', plain_exp_bs)
    # print("Running plain_exp_lr")
    # run_script('train.py', plain_exp_lr)
    # print("Running plain_exp_lr2")
    # run_script('train.py', plain_exp_lr2)
    # TODO from here
    # print("Running plain_exp_lr2_longer")
    # run_script('train.py', plain_exp_lr2_longer)

    # print("Running plain_noatt_attention_layers_removed")
    # run_script('train.py', plain_noatt_attention_layers_removed)

    print("Running plain_exp_lr2_bert")
    run_script('train.py', plain_exp_lr2_bert)

if __name__ == "__main__":
    main()
