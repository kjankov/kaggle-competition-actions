import os
import sys
import json
from pathlib import Path
from argparse import ArgumentParser
from pprint import pprint
from termcolor import colored, cprint
from zipfile import ZipFile
import pandas as pd
import subprocess
import shutil
import kaggle

from model import KaggleAction


DATASET_DIR = Path('dataset')
LEADERBOARD_DIR = Path('leaderboard')
CONFIG_PATH = Path("config.json")


def to_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def unzip(filename, parent='.'):
    with ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(parent)
    os.remove(filename)


def run_command(command, verbose=True):
    print("Executing:", command)
    p = subprocess.Popen(
        [command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = p.communicate()

    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")  # stderr contains warnings.

    if p.returncode != os.EX_OK:
        print("Return code:", p.returncode)
        print(stdout)
        print(stderr)
        raise sys.exit(p.returncode)

    if verbose:
        if stdout != "":
            print(stdout)

        if stderr != "":
            print(stderr)

    return p.returncode


def download(kapi, competition):
    cprint('\nDownloading...', attrs=['bold'])
    kapi.competition_download_files(competition=competition, path='./dataset')
    cprint('Download Complete!\n', color='green', attrs=['bold'])

    cprint('Unzipping...', attrs=['bold'])
    filename = 'dataset/{}.zip'.format(competition)
    with ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('dataset')
    os.remove(filename)
    return list(DATASET_DIR.iterdir())


def list_files(kapi, competition):
    competition_files = list(kapi.competition_list_files(competition=competition))
    cprint(text='Competition Files:  {}'.format(competition_files), attrs=['bold'])
    return competition_files


def download_leaderboard(kapi, competition, show_n=20):
    cprint(text='Downloading Leaderboard...', color='yellow')
    kapi.competition_leaderboard_download(competition=competition, path='./leaderboard')
    unzip('leaderboard/{}.zip'.format(competition), parent='leaderboard')


def view_leaderboard(kapi, competition, show_n=20):
    ld_filename = list(Path('leaderboard/').glob('*.csv'))[0]
    board = pd.read_csv(ld_filename)
    print(board.head(show_n))

    username = kapi.read_config_file()['username']
    if username in board['TeamName']:
        print('...')
        print(colored('Your Rank:', attrs=['underline']), sep=" ")
        print(board['TeamName' == username])


def submit(kapi, competition, message='Submit'):
    cprint(text='Submitting...', color='magenta')
    kapi.competition_submit(file_name='results.csv', message=message, competition=competition)
    cprint(text='Submitted successfully.\n', color='green')


def list_submissions(kapi, competition):
    submissions = kapi.competitions_submissions_list(id=competition)
    if len(submissions) > 0:
        cprint(text='Your Submissions: ', color='magenta')
        submissions_df = pd.DataFrame(submissions).sort_values(by='date')
        print(submissions_df.head())
        scores = submissions_df['publicScore'].sort_values()
        print('Your High Score:  {}\n'.format(scores[0]))
        return submissions_df
    else:
        cprint('You have no submissions for Competition {}\n'.format(competition), color='red', attrs=['bold'])


def search(keyword=None):
    if keyword is not None:
        return kaggle.api.competitions_list(search=keyword)
    else:
        return kaggle.api.competitions_list()


def main():
    kapi = kaggle.KaggleApi()
    kapi.authenticate()

    config = kapi.read_config_file()
    if 'competition' not in config:
        competitions = []
        found = False

        while not found:
            while len(competitions) == 0:
                keyword = input("Search competitions [KEYWORD]: ")
                if len(keyword) > 0:
                    competitions = search(keyword)
                else:
                    print("No results... try again.")

            print("Which competition would you like? [Select number]")
            for idx, comp in enumerate(competitions):
                print('  {})  {}'.format(idx+1, comp))
            comp_no = int(input('[SELECTION]: '))

            if comp_no - 1 < len(competitions):
                competition = competitions[comp_no - 1]
                kapi.set_config_value(name='competition', value=competition)
                found = True
            else:
                print('Invalid number.')
    else:
        competition = config['competition']

    datafiles = [f.parts[-1] for f in DATASET_DIR.iterdir()]
    if len(datafiles) == 0:
        competition_files = list_files(kapi, competition)
        download(kapi, competition)
    else:
        competition_files = datafiles

    gt_file = [f for f in datafiles if (f != 'train.csv' and f != 'test.csv')]
    gt_file = 'dataset/{}'.format(gt_file[0]) if len(gt_file) > 0 else None
    kaction = KaggleAction(train_filename='dataset/train.csv',
                           test_filename='dataset/test.csv',
                           groundtruth_filename=gt_file)
    kaction.train()
    kaction.test()

    list_submissions(kapi, competition)

    if len(list(LEADERBOARD_DIR.iterdir())) == 0:
        download_leaderboard(kapi, competition)
    view_leaderboard(kapi, competition)

    answer = False
    while not answer:
        choice = input(colored('Would you like to submit this result? [Y/n]: ', attrs=['bold']))
        choice = choice.upper()
        if choice == 'Y' or choice == 'YES':
            submit(kapi, competition)
            answer = True
        elif choice == 'N' or choice == 'NO':
            answer = True
        else:
            print('Input not understood. Try again.')


if __name__ == "__main__":
    main()