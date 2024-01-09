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

    # Call the train.py script with the specified arguments
    print("Running plain_att")
    run_script('train.py', plain_att)
    print("Running plain_noatt")
    run_script('train.py', plain_noatt)
    print("Running bert_att")
    run_script('train.py', bert_att)
    print("Running bert_noatt")
    run_script('train.py', bert_noatt)

if __name__ == "__main__":
    main()
