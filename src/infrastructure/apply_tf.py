import subprocess

def deploy_with_terraform():
    subprocess.run(['terraform', 'init'], cwd='./terraform')
    subprocess.run(['terraform', 'apply', '-auto-approve'], cwd='./terraform')