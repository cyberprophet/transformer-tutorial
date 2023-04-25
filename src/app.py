import requests
import dotenv
import os

import pandas as pd

dotenv.load_dotenv()

base_url = os.getenv('BASE')
chart_route = os.getenv('CHART')
option_route = os.getenv('DAY')

code = '005930'
period = 128

res = requests.get(
    f'https://{base_url}/{chart_route}/{option_route}?code={code}&period={period}')

df = pd.json_normalize(res.json()['chart'])
