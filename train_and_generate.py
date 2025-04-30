import pandas as pd
from synthcity.utils.serialization import save_to_file, load_from_file
from synthcity.plugins import Plugins
import argparse
from synthcity.plugins.core.dataloader import GenericDataLoader


parser = argparse.ArgumentParser(description="Generate TVAE data",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", required=True, type=str, help="Input file")
parser.add_argument("-o", "--output", required=False, type=str, help="Output file")

args = parser.parse_args()
input_file = args.input
output_file = args.output

df = pd.read_json(f"data/{input_file}.json")
string_cols = ["coach_id", "player_id", "club_id", "season_id",
               "validity_start_year", "validity_start_month", "validity_start_day",
               "validity_end_year", "validity_end_month", "validity_end_day",
               "date_of_birth_year", "date_of_birth_month", "date_of_birth_day",
               "reason_injury", "reason_injury_end", "reason_new_coach", "reason_market_value_update",
               "reason_transfer", "reason_regular_interval"]

df[string_cols] = df[string_cols].astype(str)
tvae = Plugins().get("tvae", n_iter=2000, n_iter_min=1000)
tvae.fit(df)

training_data_sample_size = df.shape[0]
samples = tvae.generate(count=training_data_sample_size)
samples_df = samples.dataframe()

samples_df.to_json(f"output/{output_file}.json", orient="records")