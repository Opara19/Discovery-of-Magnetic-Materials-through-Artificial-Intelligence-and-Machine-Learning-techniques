import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import folium 
from streamlit_folium import folium_static
import json
from matplotlib import cm, colors
from streamlit_option_menu import option_menu
from PIL import Image
import altair as alt
from datetime import datetime
import time
import pickle
import pydeck as pdk
import pymatgen
from mp_api.client import MPRester
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline

from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from matminer.featurizers.composition.ion import OxidationStates
from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition
from matminer.featurizers.composition import (ElectronegativityDiff, ElectronAffinity, BandCenter, Stoichiometry,
                                              Meredig, ElementProperty)
from matminer.featurizers.composition.alloy import YangSolidSolution
from matminer.featurizers.composition.orbital import ValenceOrbital, AtomicOrbitals
import joblib

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

API_KEY = 'zPxv3pPc0Z14GQDyz3M2GNK3IIJNQBB2'

coln1,coln2=st.columns([0.8, 3.2])

with coln1:
    st.image("logo2.png",width=500)
with coln2:
     st.markdown("<h1 style='font-size: 80px; color: white;'>MaGpRoPs</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 30px; color: white;'>Material Properties Prediction</h1>", unsafe_allow_html=True)
#st.title("Material Properties Prediction ")
st.write("")
st.write("")


formula_input = st.text_input("Enter the formula of your material 👩‍🔬: ", placeholder="e.g., Fe2O3")
st.write("")
st.write("")
st.markdown(
    """
    <style>
        /* Apply background color to the main page */
        .stApp {
            background-color: #070c21;  /* Light grayish blue */
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.write("▫️ Use below Extract Features button to extract the features needed")
if st.button("Extract Features"):
    if formula_input:
        try:
            # Convert formula to composition
            composition = Composition(formula_input)
            df = pd.DataFrame({'composition': [composition]})
            
            # Oxidation States
            try:
                oxidation_states = OxidationStates()
                df = oxidation_states.featurize_dataframe(df, col_id="composition", ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during oxidation states featurization: {e}")

            # Convert to oxidation composition
            try:
                cf = CompositionToOxidComposition(target_col_id='oxidation_composition')
                df = cf.featurize_dataframe(df, 'composition', ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during oxidation composition conversion: {e}")

            # Elemental Properties
            try:
                electronegativity_diff = ElectronegativityDiff()
                df = electronegativity_diff.featurize_dataframe(df, col_id="oxidation_composition", ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during electronegativity diff featurization: {e}")

            try:
                electron_affinity = ElectronAffinity(impute_nan=True)
                df = electron_affinity.featurize_dataframe(df, col_id="oxidation_composition", ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during electron affinity featurization: {e}")

            try:
                valence_orbital = ValenceOrbital(impute_nan=True)
                df = valence_orbital.featurize_dataframe(df, col_id="composition", ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during valence orbital featurization: {e}")

            try:
                band_center = BandCenter(impute_nan=True)
                df = band_center.featurize_dataframe(df, col_id="composition", ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during band center featurization: {e}")

            try:
                atomic_orbital = AtomicOrbitals()
                df = atomic_orbital.featurize_dataframe(df, col_id="composition", ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during atomic orbital featurization: {e}")

            try:
                stoichiometry = Stoichiometry()
                df = stoichiometry.featurize_dataframe(df, col_id="composition", ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during stoichiometry featurization: {e}")
            df.drop(['avg s valence electrons','avg p valence electrons','avg d valence electrons',
                    'avg f valence electrons','frac s valence electrons','frac p valence electrons',
                    'frac d valence electrons','frac f valence electrons'],axis=1,inplace=True)
            
            try:
                meredig = Meredig(impute_nan=True)
                df = meredig.featurize_dataframe(df, col_id="composition", ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during Meredig featurization: {e}")
            df.drop(["He fraction","Ne fraction","Ar fraction","Kr fraction",
                    "Yb fraction","Po fraction","At fraction","Rn fraction",
                    "Fr fraction","Ra fraction","Am fraction",
                    "Cm fraction","Bk fraction","Cf fraction","Es fraction",
                    "Fm fraction","Md fraction","No fraction","Lr fraction",
                    "Rf fraction","Db fraction","Sg fraction","Bh fraction",
                    "Hs fraction","Mt fraction","Ds fraction","Rg fraction",
                    "Cn fraction","Nh fraction","Fl fraction","Mc fraction",
                    "Lv fraction","Ts fraction","Og fraction"],
                    axis=1, inplace=True)
            try:
                element_property_magpie = ElementProperty.from_preset('magpie',impute_nan=True)
                df = element_property_magpie.featurize_dataframe(df, col_id='composition', ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during magpie featurization: {e}")

            try:
                yang_solid_solution = YangSolidSolution(impute_nan=True)
                df = yang_solid_solution.featurize_dataframe(df, col_id="composition", ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during Yang Solid Solution featurization: {e}")

            try:
                element_property_megnet = ElementProperty.from_preset('megnet_el',impute_nan=True)
                df = element_property_megnet.featurize_dataframe(df, col_id='composition', ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during megnet_el featurization: {e}")

            try:
                element_property_matminer = ElementProperty.from_preset('matminer',impute_nan=True)
                df = element_property_matminer.featurize_dataframe(df, col_id='composition', ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during matminer featurization: {e}")
            df.drop(['PymatgenData minimum X',
                'PymatgenData maximum X',
                'PymatgenData range X',
                'PymatgenData mean X',
                'PymatgenData std_dev X',
                'PymatgenData minimum row',
                'PymatgenData maximum row',
                'PymatgenData range row',
                'PymatgenData mean row',
                'PymatgenData std_dev row','PymatgenData minimum atomic_mass',
                'PymatgenData maximum atomic_mass',
                'PymatgenData range atomic_mass',
                'PymatgenData mean atomic_mass',
                'PymatgenData std_dev atomic_mass',
                'PymatgenData minimum atomic_radius',
                'PymatgenData maximum atomic_radius',
                'PymatgenData range atomic_radius',
                'PymatgenData mean atomic_radius',
                'PymatgenData std_dev atomic_radius',
                'PymatgenData minimum mendeleev_no',
                'PymatgenData maximum mendeleev_no',
                'PymatgenData range mendeleev_no',
                'PymatgenData mean mendeleev_no',
                'PymatgenData std_dev mendeleev_no','PymatgenData minimum melting_point',
                'PymatgenData maximum melting_point',
                'PymatgenData range melting_point',
                'PymatgenData mean melting_point',
                'PymatgenData std_dev melting_point','PymatgenData minimum coefficient_of_linear_thermal_expansion',
                'PymatgenData maximum coefficient_of_linear_thermal_expansion',
                'PymatgenData range coefficient_of_linear_thermal_expansion',
                'PymatgenData mean coefficient_of_linear_thermal_expansion',
                'PymatgenData std_dev coefficient_of_linear_thermal_expansion'],axis=1,inplace=True)
            try:
                element_property_deml = ElementProperty.from_preset('deml',impute_nan=True)
                df = element_property_deml.featurize_dataframe(df, col_id='composition', ignore_errors=True)
            except Exception as e:
                st.warning(f"Error during DEML featurization: {e}")
            df.drop(['DemlData minimum atom_num',
                'DemlData maximum atom_num',
                'DemlData range atom_num',
                'DemlData mean atom_num',
                'DemlData std_dev atom_num','DemlData minimum atom_mass',
                'DemlData maximum atom_mass',
                'DemlData range atom_mass',
                'DemlData mean atom_mass',
                'DemlData std_dev atom_mass','DemlData minimum row_num',
                'DemlData maximum row_num',
                'DemlData range row_num',
                'DemlData mean row_num',
                'DemlData std_dev row_num',
                'DemlData minimum col_num',
                'DemlData maximum col_num',
                'DemlData range col_num',
                'DemlData mean col_num',
                'DemlData std_dev col_num',
                'DemlData minimum atom_radius',
                'DemlData maximum atom_radius',
                'DemlData range atom_radius',
                'DemlData mean atom_radius',
                'DemlData std_dev atom_radius','DemlData minimum melting_point',
                'DemlData maximum melting_point',
                'DemlData range melting_point',
                'DemlData mean melting_point',
                'DemlData std_dev melting_point','DemlData minimum electronegativity',
                'DemlData maximum electronegativity',
                'DemlData range electronegativity',
                'DemlData mean electronegativity',
                'DemlData std_dev electronegativity'],axis=1,inplace=True)
            
            
            df.drop(['minimum oxidation state','maximum oxidation state', 'range oxidation state','std_dev oxidation state'],axis=1,inplace=True)
            # Display DataFrame
            st.session_state["features_df"] = df
            st.write("Extracted Features:", df)
            st.write("Extracted Features:", df.shape)
            #st.write("Extracted Features:", df.columns.to_list())
        except Exception as e:
            st.error(f"Error processing formula: {e}")

st.write("")
st.write("")
st.write("▫️ Click on any of the buttons below to predict the properties of the material")

#model_temp= joblib.load("temp_pred_stack_model.pkl")
# Magnetism prediction
if st.button("Predict Magnetism"):
    model_mag = joblib.load("magnetism_pred_xgb_model.pkl")
    if "features_df" in st.session_state:  # Ensure df is available
        try:
            df = st.session_state["features_df"]
            df_magnetism=df.drop(['composition','oxidation_composition','MagpieData minimum NpValence', 'MagpieData minimum NpUnfilled',
                   'MagpieData minimum GSbandgap', 'PymatgenData maximum block','MagpieData minimum GSmagmom',
             'MagpieData maximum GSmagmom','MagpieData range GSmagmom','MagpieData mean GSmagmom',
             'MagpieData avg_dev GSmagmom','MagpieData mode GSmagmom','MagpieData range NfUnfilled',
             'DemlData range heat_cap','MagpieData maximum NfUnfilled','DemlData mean molar_vol',
             'MagpieData minimum NfUnfilled','MEGNetElementData mean embedding 12','DemlData maximum molar_vol',
             'MEGNetElementData range embedding 12','MagpieData avg_dev MendeleevNumber',
             'PymatgenData minimum electrical_resistivity','Xe fraction','Ba fraction',
             'MagpieData range NdValence','range EN difference','W fraction','DemlData std_dev heat_cap',
             'MEGNetElementData mean embedding 16','Re fraction','Ru fraction','Se fraction','Cl fraction',
             'PymatgenData range electrical_resistivity','Br fraction','Hg fraction','PymatgenData mean electrical_resistivity',
             'Hf fraction','Sr fraction','Ir fraction','MagpieData minimum NUnfilled',
             'I fraction','Pb fraction','Rb fraction','Be fraction','Cd fraction','Lu fraction',
             'Bi fraction','Te fraction','Cs fraction','Zr fraction','Ca fraction','As fraction','Ac fraction',
             'MagpieData minimum NdUnfilled','Pm fraction','Au fraction','K fraction','Tl fraction',
             'Np fraction','10-norm','3-norm','5-norm','7-norm','DemlData maximum first_ioniz',
             'DemlData maximum mus_fere','DemlData mean boiling_point','DemlData mean first_ioniz',
             'DemlData mean heat_fusion','DemlData mean mus_fere','DemlData minimum boiling_point',
             'DemlData minimum first_ioniz','DemlData minimum heat_fusion','DemlData minimum mus_fere',
             'DemlData range boiling_point','DemlData range electric_pol','DemlData range first_ioniz','DemlData range heat_fusion',
             'DemlData range molar_vol','DemlData range mus_fere',
             'DemlData std_dev FERE correction','DemlData std_dev boiling_point','DemlData std_dev electric_pol','DemlData std_dev first_ioniz',
             'DemlData std_dev heat_fusion','DemlData std_dev molar_vol','DemlData std_dev mus_fere','LUMO_energy','MEGNetElementData maximum embedding 16',
             'MEGNetElementData maximum embedding 6','MEGNetElementData mean embedding 10','MEGNetElementData mean embedding 11','MEGNetElementData mean embedding 14','MEGNetElementData mean embedding 15',
             'MEGNetElementData mean embedding 4','MEGNetElementData mean embedding 5','MEGNetElementData mean embedding 6','MEGNetElementData mean embedding 7',
             'MEGNetElementData mean embedding 8','MEGNetElementData mean embedding 9','MEGNetElementData minimum embedding 10','MEGNetElementData minimum embedding 11',
             'MEGNetElementData minimum embedding 7','MEGNetElementData minimum embedding 8','MEGNetElementData range embedding 1','MEGNetElementData range embedding 10',
             'MEGNetElementData range embedding 11','MEGNetElementData range embedding 13','MEGNetElementData range embedding 2','MEGNetElementData range embedding 8',
             'MEGNetElementData range embedding 9','MEGNetElementData std_dev embedding 1','MEGNetElementData std_dev embedding 10','MEGNetElementData std_dev embedding 11',
             'MEGNetElementData std_dev embedding 13','MEGNetElementData std_dev embedding 14','MEGNetElementData std_dev embedding 15','MEGNetElementData std_dev embedding 16',
             'MEGNetElementData std_dev embedding 2','MEGNetElementData std_dev embedding 3','MEGNetElementData std_dev embedding 4','MEGNetElementData std_dev embedding 5',
             'MEGNetElementData std_dev embedding 6','MEGNetElementData std_dev embedding 7','MEGNetElementData std_dev embedding 8','MEGNetElementData std_dev embedding 9',
             'MagpieData avg_dev AtomicWeight','MagpieData avg_dev CovalentRadius','MagpieData avg_dev Electronegativity','MagpieData avg_dev GSbandgap',
             'MagpieData avg_dev GSvolume_pa','MagpieData avg_dev NUnfilled','MagpieData avg_dev NdUnfilled','MagpieData avg_dev NfUnfilled','MagpieData avg_dev NfValence',
             'MagpieData avg_dev NpValence','MagpieData avg_dev NsUnfilled','MagpieData avg_dev NsValence','MagpieData avg_dev Row','MagpieData avg_dev SpaceGroupNumber',
             'MagpieData maximum AtomicWeight','MagpieData maximum Column','MagpieData maximum Electronegativity','MagpieData maximum NfValence','MagpieData maximum NpValence',
             'MagpieData maximum NsUnfilled','MagpieData maximum Row','MagpieData mean AtomicWeight','MagpieData mean Column','MagpieData mean CovalentRadius',
             'MagpieData mean Electronegativity','MagpieData mean GSbandgap','MagpieData mean MeltingT','MagpieData mean NValence','MagpieData mean NdValence','MagpieData mean NfValence','MagpieData mean NpValence',
             'MagpieData mean NsUnfilled',
             'MagpieData mean NsValence',
             'MagpieData mean Number',
             'MagpieData mean Row',
             'MagpieData mean SpaceGroupNumber',
             'MagpieData minimum AtomicWeight',
             'MagpieData minimum CovalentRadius',
             'MagpieData minimum Electronegativity',
             'MagpieData minimum NsUnfilled',
             'MagpieData minimum Row',
             'MagpieData minimum SpaceGroupNumber',
             'MagpieData mode AtomicWeight',
             'MagpieData mode Column',
             'MagpieData mode CovalentRadius',
             'MagpieData mode Electronegativity',
             'MagpieData mode GSbandgap',
             'MagpieData mode MeltingT',
             'MagpieData mode NdValence',
             'MagpieData mode NpValence',
             'MagpieData mode Number',
             'MagpieData mode Row',
             'MagpieData mode SpaceGroupNumber',
             'MagpieData range AtomicWeight',
             'MagpieData range Column',
             'MagpieData range CovalentRadius',
             'MagpieData range Electronegativity',
             'MagpieData range GSbandgap',
             'MagpieData range GSvolume_pa',
             'MagpieData range MendeleevNumber',
             'MagpieData range NUnfilled',
             'MagpieData range NValence',
             'MagpieData range NdUnfilled',
             'MagpieData range NfValence',
             'MagpieData range NpUnfilled',
             'MagpieData range NpValence',
             'MagpieData range NsUnfilled',
             'MagpieData range NsValence',
             'MagpieData range Row',
             'MagpieData range SpaceGroupNumber',
             'PymatgenData maximum electrical_resistivity',
             'PymatgenData maximum group',
             'PymatgenData mean block',
             'PymatgenData mean group',
             'PymatgenData minimum group',
             'PymatgenData range block',
             'PymatgenData range bulk_modulus',
             'PymatgenData range group',
             'PymatgenData range thermal_conductivity',
             'PymatgenData range velocity_of_sound',
             'PymatgenData std_dev bulk_modulus',
             'PymatgenData std_dev electrical_resistivity',
             'PymatgenData std_dev group',
             'PymatgenData std_dev thermal_conductivity',
             'PymatgenData std_dev velocity_of_sound',
             'Yang delta',
             'avg p valence electrons',
             'frac d valence electrons',
             'frac f valence electrons',
             'frac p valence electrons',
             'mean AtomicRadius',
             'mean EN difference','mean AtomicWeight',
             'mean Column',
             'mean Row',
             'range Number',
             'mean Number','range Electronegativity',
             'mean Electronegativity'],axis=1)
            df_magnetism = df_magnetism.rename(columns={'HOMO_element': 'HOMO_element_imputed', 'LUMO_element': 'LUMO_element_imputed','LUMO_character': 'LUMO_character_imputed','HOMO_character': 'HOMO_character_imputed'})
            
            # Separate categorical and numerical columns
            categorical_cols = df_magnetism.select_dtypes(include=['object']).columns
            
            
            
            df_magnetism = df_magnetism.rename(columns={
                'H fraction': 'H', 'Li fraction': 'Li', 'B fraction': 'B', 'C fraction': 'C', 'N fraction': 'N',
                'O fraction': 'O', 'F fraction': 'F', 'Na fraction': 'Na', 'Mg fraction': 'Mg', 'Al fraction': 'Al',
                'Si fraction': 'Si', 'P fraction': 'P', 'S fraction': 'S', 'Sc fraction': 'Sc', 'Ti fraction': 'Ti',
                'V fraction': 'V', 'Cr fraction': 'Cr', 'Mn fraction': 'Mn', 'Fe fraction': 'Fe', 'Co fraction': 'Co',
                'Ni fraction': 'Ni', 'Cu fraction': 'Cu', 'Zn fraction': 'Zn', 'Ga fraction': 'Ga', 'Ge fraction': 'Ge',
                'Y fraction': 'Y', 'Nb fraction': 'Nb', 'Mo fraction': 'Mo', 'Tc fraction': 'Tc', 'Rh fraction': 'Rh',
                'Pd fraction': 'Pd', 'Ag fraction': 'Ag', 'In fraction': 'In', 'Sn fraction': 'Sn', 'Sb fraction': 'Sb',
                'La fraction': 'La', 'Ce fraction': 'Ce', 'Pr fraction': 'Pr', 'Nd fraction': 'Nd', 'Sm fraction': 'Sm',
                'Eu fraction': 'Eu', 'Gd fraction': 'Gd', 'Tb fraction': 'Tb', 'Dy fraction': 'Dy', 'Ho fraction': 'Ho',
                'Er fraction': 'Er', 'Tm fraction': 'Tm', 'Ta fraction': 'Ta', 'Os fraction': 'Os', 'Pt fraction': 'Pt',
                'Th fraction': 'Th','Pa fraction': 'Pa', 'U fraction': 'U', 'Pu fraction': 'Pu'})
           # st.write(df_magnetism.columns)     
            with open('magnetism_feature_names.pkl', 'rb') as f:
                    expected_feature_names = pickle.load(f)
            
            
            def prepare_data_for_prediction(df):
                processed_df = df.copy()
                categorical_mappings = {
                    'HOMO_element_imputed':["Ag", "Al", "As", "Au", "B", "Ba", "Be", "Bi", "Br", "C", "Ca", "Cd", "Ce", "Cl", "Co", "Cr", "Cs",  
                            "Cu", "Dy", "Er", "Eu", "F", "Fe", "Ga", "Gd", "Ge", "H", "Hf", "Hg", "Ho", "I", "In", "Ir", "K",  
                            "La", "Li", "Lu", "Mg", "Mn", "Mo", "N", "Na", "Nb", "Nd", "Ni", "Np", "O", "Os", "P", "Pa", "Pb",  
                            "Pd", "Pm", "Pr", "Pt", "Pu", "Rb", "Re", "Rh", "Ru", "S", "Sb", "Sc", "Se", "Si", "Sm", "Sn", "Sr",  
                            "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl", "Tm", "U", "V", "W", "Xe", "Y", "Zn", "Zr"
                    ],
                    'LUMO_element_imputed':[ "Ag", "Al", "As", "Au", "B", "Ba", "Be", "Bi", "Br", "C", "Ca", "Cd", "Ce", "Cl", "Co", "Cr", "Cs",
                            "Cu", "Dy", "Er", "Eu", "F", "Fe", "Ga", "Gd", "Ge", "H", "Hf", "Hg", "Ho", "I", "In", "Ir", "K", "La", "Li",
                            "Lu", "Mg", "Mn", "Mo", "N", "Na", "Nb", "Nd", "Ni", "Np", "O", "Os", "P", "Pa", "Pb", "Pd", "Pm", "Pr", "Pt",
                            "Pu", "Rb", "Re", "Rh", "Ru", "S", "Sb", "Sc", "Se", "Si", "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te", "Th", "Ti",
                            "Tl", "Tm", "U", "V", "W", "Xe", "Y", "Zn", "Zr"],
                    'LUMO_character_imputed': ['p', 's','f'],
                    'HOMO_character_imputed':['f','s','p'],
                            
                }
                for column, values in categorical_mappings.items():
                    for value in values:
                        column_name = f"{column}_{value}"
                        processed_df[column_name] = (processed_df[column] == value).astype(int)
                
                    processed_df = processed_df.drop(column, axis=1)

                
                

                processed_df = processed_df.reindex(columns=expected_feature_names, fill_value=0)

                #processed_df
                X = processed_df.copy()

                
                return X
            #st.write(prepare_data_for_prediction(df_magnetism).columns.to_list())
            
            
            
            # def compare_features(df, feature_names):
            #     # Get the columns in df_magnetism
            #     df_columns = set(df.columns)
                
            #     # Get the feature names from the model
            #     model_features = set(feature_names)
                
            #     # Find missing features in df_magnetism
            #     missing_features = model_features - df_columns
                
            #     # Find extra features in df_magnetism
            #     extra_features = df_columns - model_features
                
            #     return missing_features, extra_features

            # # Compare features
            # missing_features, extra_features = compare_features(prepare_data_for_prediction(df_magnetism), expected_feature_names)

            # # Print results
            # if missing_features:
            #     st.write("Missing features in df_magnetism:", missing_features)
            # else:
            #     st.write("No missing features in df_magnetism.")

            # if extra_features:
            #     st.write("Extra features in df_magnetism:", extra_features)
            # else:
            #     st.write("No extra features in df_magnetism.")
            numerical_cols = df_magnetism.select_dtypes(exclude=['object']).columns
            #st.write(df_magnetism[numerical_cols].mean())
            #st.write(df_magnetism[numerical_cols].std())
            # st.write(df_magnetism[numerical_cols].dtypes)
            # for col in numerical_cols:
            #     st.write(f"Column: {col}, Std Dev: {df_magnetism[col].std()}")
            class StdScaler(BaseEstimator, TransformerMixin):
                def __init__(self, columns=None):
                    self.columns = columns
                    self.scaler = StandardScaler()

                def fit(self, X, y=None):
                    if (self.columns is None):
                        num_cols = [i for i in X.columns if len(np.unique(X[i])) > 5]
                        self.columns = num_cols
                    self.scaler.fit(X[self.columns])
                    return self

                def transform(self, X, y=None):
                    scaled_data = self.scaler.transform(X[self.columns])
                    result_df = pd.DataFrame(scaled_data, columns=self.columns, index=X.index)
                    return pd.concat([X.drop(columns=self.columns), result_df], axis=1)
            # scaler=StdScaler(columns=numerical_cols)
            # df_magnetism=scaler.fit_transform(df_magnetism)
            # df_magnetism
            scaling_pipeline=joblib.load("scalingpipe.pkl")
            df_magnetism=scaling_pipeline.transform(df_magnetism)
            #df_magnetism
            predictions = model_mag.predict(prepare_data_for_prediction(df_magnetism))
            st.session_state["mag_status"]=predictions
            # Display results
            if(predictions==1):
                st.markdown("<h1 style='font-size: 18px; color: red;'>Is a magnetic material</h1>", unsafe_allow_html=True)
                # st.write("Is a magnetic material")
            else:
                st.markdown("<h1 style='font-size: 18px; color: red;'>Not a magnetic material</h1>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please extract features first before predicting.")


# Ordering prediction
if st.button("Predict Magnetic Ordering"):
    model_order= joblib.load("ordering_pred_xgb_model.pkl")
    if "features_df" in st.session_state:  # Ensure df is available
        try:
            df = st.session_state["features_df"]
            df_mag_order=df.drop(['composition','oxidation_composition','MagpieData minimum NpValence', 'MagpieData minimum NpUnfilled',
                    'MagpieData minimum GSbandgap', 'PymatgenData maximum block','Pm fraction',
                    'DemlData minimum FERE correction','C fraction','MagpieData range NdUnfilled',
                    'Xe fraction', 'Ac fraction','Si fraction','Pa fraction',
                    'MagpieData maximum NsValence','MagpieData minimum NfUnfilled','MEGNetElementData minimum embedding 9',
                    'As fraction','MagpieData minimum NsUnfilled','MagpieData minimum GSmagmom','S fraction',
                    'PymatgenData maximum electrical_resistivity','PymatgenData range electrical_resistivity',
                    'PymatgenData mean electrical_resistivity','PymatgenData std_dev electrical_resistivity','Ti fraction',
                    'Pd fraction',
                    'MagpieData mode NfUnfilled',
                    'DemlData std_dev mus_fere',
                    'Cd fraction',
                    'Y fraction',
                    'Au fraction',
                    'Ge fraction',
                    'Th fraction',
                    'Hg fraction',
                    'Sn fraction',
                    'MagpieData minimum NfValence',
                    'Ce fraction',
                    'MEGNetElementData mean embedding 16',
                    'Bi fraction',
                    'Zr fraction',
                    'Hf fraction',
                    'DemlData mean FERE correction',
                    'MEGNetElementData mean embedding 6',
                    'MEGNetElementData std_dev embedding 4',
                    'MEGNetElementData mean embedding 3',
                    'Ir fraction',
                    'MEGNetElementData std_dev embedding 1',
                    'MEGNetElementData mean embedding 1',
                    'MagpieData avg_dev GSvolume_pa',
                    'Ga fraction',
                    'DemlData std_dev GGAU_Etot',
                    'MagpieData mean GSvolume_pa',
                    'K fraction',
                    'MagpieData mode NsUnfilled',
                    'MEGNetElementData mean embedding 13',
                    'DemlData mean electric_pol',
                    'MEGNetElementData mean embedding 14',
                    'Pb fraction',
                    'MEGNetElementData std_dev embedding 14',
                    'MagpieData avg_dev MeltingT',
                    'MagpieData avg_dev AtomicWeight',
                    'Dy fraction',
                    'Be fraction',
                    'MEGNetElementData std_dev embedding 15',
                    'gap_AO',
                    'Np fraction',
                    'MEGNetElementData mean embedding 9','10-norm',
                    '3-norm',
                    '5-norm',
                    '7-norm',
                    'DemlData maximum first_ioniz',
                    'DemlData maximum molar_vol',
                    'DemlData maximum mus_fere',
                    'DemlData mean boiling_point',
                    'DemlData mean first_ioniz',
                    'DemlData mean mus_fere',
                    'DemlData minimum boiling_point',
                    'DemlData minimum first_ioniz',
                    'DemlData minimum heat_fusion',
                    'DemlData minimum mus_fere',
                    'DemlData range boiling_point',
                    'DemlData range electric_pol',
                    'DemlData range first_ioniz',
                    'DemlData range heat_fusion',
                    'DemlData range molar_vol',
                    'DemlData range mus_fere',
                    'DemlData std_dev FERE correction',
                    'DemlData std_dev boiling_point',
                    'DemlData std_dev electric_pol',
                    'DemlData std_dev first_ioniz',
                    'DemlData std_dev heat_cap',
                    'DemlData std_dev heat_fusion',
                    'DemlData std_dev molar_vol',
                    'LUMO_energy',
                    'MEGNetElementData maximum embedding 10',
                    'MEGNetElementData maximum embedding 6',
                    'MEGNetElementData mean embedding 10',
                    'MEGNetElementData mean embedding 11',
                    'MEGNetElementData mean embedding 15',
                    'MEGNetElementData mean embedding 5',
                    'MEGNetElementData mean embedding 7',
                    'MEGNetElementData mean embedding 8',
                    'MEGNetElementData minimum embedding 10',
                    'MEGNetElementData minimum embedding 11',
                    'MEGNetElementData minimum embedding 7',
                    'MEGNetElementData minimum embedding 8',
                    'MEGNetElementData range embedding 1',
                    'MEGNetElementData range embedding 10',
                    'MEGNetElementData range embedding 11',
                    'MEGNetElementData range embedding 8',
                    'MEGNetElementData std_dev embedding 10',
                    'MEGNetElementData std_dev embedding 11',
                    'MEGNetElementData std_dev embedding 12',
                    'MEGNetElementData std_dev embedding 13',
                    'MEGNetElementData std_dev embedding 16',
                    'MEGNetElementData std_dev embedding 2',
                    'MEGNetElementData std_dev embedding 3',
                    'MEGNetElementData std_dev embedding 5',
                    'MEGNetElementData std_dev embedding 6',
                    'MEGNetElementData std_dev embedding 7',
                    'MEGNetElementData std_dev embedding 8',
                    'MEGNetElementData std_dev embedding 9',
                    'MagpieData avg_dev Electronegativity',
                    'MagpieData avg_dev GSbandgap',
                    'MagpieData avg_dev GSmagmom',
                    'MagpieData avg_dev NUnfilled',
                    'MagpieData avg_dev NfUnfilled',
                    'MagpieData avg_dev NfValence',
                    'MagpieData avg_dev NpValence',
                    'MagpieData avg_dev NsUnfilled',
                    'MagpieData avg_dev NsValence',
                    'MagpieData avg_dev Row',
                    'MagpieData avg_dev SpaceGroupNumber',
                    'MagpieData maximum AtomicWeight',
                    'MagpieData maximum Column',
                    'MagpieData maximum Electronegativity',
                    'MagpieData maximum NUnfilled',
                    'MagpieData maximum NfValence',
                    'MagpieData maximum NpValence',
                    'MagpieData maximum NsUnfilled',
                    'MagpieData maximum Row',
                    'MagpieData mean AtomicWeight',
                    'MagpieData mean Column',
                    'MagpieData mean CovalentRadius',
                    'MagpieData mean Electronegativity',
                    'MagpieData mean GSbandgap',
                    'MagpieData mean GSmagmom',
                    'MagpieData mean MeltingT',
                    'MagpieData mean NValence',
                    'MagpieData mean NdValence',
                    'MagpieData mean NfValence',
                    'MagpieData mean NpValence',
                    'MagpieData mean NsUnfilled',
                    'MagpieData mean NsValence',
                    'MagpieData mean Number',
                    'MagpieData mean Row',
                    'MagpieData mean SpaceGroupNumber',
                    'MagpieData minimum AtomicWeight',
                    'MagpieData minimum CovalentRadius',
                    'MagpieData minimum Electronegativity',
                    'MagpieData minimum Row',
                    'MagpieData minimum SpaceGroupNumber',
                    'MagpieData mode AtomicWeight',
                    'MagpieData mode Column',
                    'MagpieData mode CovalentRadius',
                    'MagpieData mode Electronegativity',
                    'MagpieData mode GSbandgap',
                    'MagpieData mode MeltingT',
                    'MagpieData mode NdValence',
                    'MagpieData mode NpValence',
                    'MagpieData mode Number',
                    'MagpieData mode Row',
                    'MagpieData mode SpaceGroupNumber',
                    'MagpieData range AtomicWeight',
                    'MagpieData range CovalentRadius',
                    'MagpieData range Electronegativity',
                    'MagpieData range GSbandgap',
                    'MagpieData range GSmagmom',
                    'MagpieData range GSvolume_pa',
                    'MagpieData range MendeleevNumber',
                    'MagpieData range NUnfilled',
                    'MagpieData range NValence',
                    'MagpieData range NdValence',
                    'MagpieData range NfUnfilled',
                    'MagpieData range NfValence',
                    'MagpieData range NpUnfilled',
                    'MagpieData range NpValence',
                    'MagpieData range NsUnfilled',
                    'MagpieData range NsValence',
                    'MagpieData range Number',
                    'MagpieData range Row',
                    'MagpieData range SpaceGroupNumber',
                    'PymatgenData maximum group',
                    'PymatgenData mean group',
                    'PymatgenData minimum group',
                    'PymatgenData range block',
                    'PymatgenData range bulk_modulus',
                    'PymatgenData range group',
                    'PymatgenData range thermal_conductivity',
                    'PymatgenData range velocity_of_sound',
                    'PymatgenData std_dev bulk_modulus',
                    'PymatgenData std_dev group',
                    'PymatgenData std_dev thermal_conductivity',
                    'PymatgenData std_dev velocity_of_sound',
                    'Yang delta',
                    'avg p valence electrons',
                    'frac d valence electrons',
                    'frac f valence electrons',
                    'frac p valence electrons',
                    'mean AtomicRadius',
                    'mean EN difference',
                    'std_dev EN difference'],axis=1)
            df_mag_order = df_mag_order.rename(columns={'HOMO_element': 'HOMO_element_imputed', 'LUMO_element': 'LUMO_element_imputed','LUMO_character': 'LUMO_character_imputed','HOMO_character': 'HOMO_character_imputed'})
            
            # Separate categorical and numerical columns
            categorical_cols = df_mag_order.select_dtypes(include=['object']).columns
            
            numerical_cols = df_mag_order.select_dtypes(exclude=['object']).columns
            
            df_mag_order = df_mag_order.rename(columns={
                'H fraction': 'H', 'Li fraction': 'Li', 'B fraction': 'B', 'C fraction': 'C', 'N fraction': 'N',
                'O fraction': 'O', 'F fraction': 'F', 'Na fraction': 'Na', 'Mg fraction': 'Mg', 'Al fraction': 'Al',
                'Si fraction': 'Si', 'P fraction': 'P', 'S fraction': 'S', 'Sc fraction': 'Sc',
                'V fraction': 'V', 'Cr fraction': 'Cr', 'Mn fraction': 'Mn', 'Fe fraction': 'Fe', 'Co fraction': 'Co',
                'Ni fraction': 'Ni', 'Cu fraction': 'Cu', 'Zn fraction': 'Zn', 'Ga fraction': 'Ga', 'Ge fraction': 'Ge',
                'Y fraction': 'Y', 'Nb fraction': 'Nb', 'Mo fraction': 'Mo', 'Tc fraction': 'Tc', 'Rh fraction': 'Rh',
                'Pd fraction': 'Pd', 'Ag fraction': 'Ag', 'In fraction': 'In', 'Sn fraction': 'Sn', 'Sb fraction': 'Sb',
                'La fraction': 'La', 'Ce fraction': 'Ce', 'Pr fraction': 'Pr', 'Nd fraction': 'Nd', 'Sm fraction': 'Sm',
                'Eu fraction': 'Eu', 'Gd fraction': 'Gd', 'Tb fraction': 'Tb', 'Dy fraction': 'Dy', 'Ho fraction': 'Ho',
                'Er fraction': 'Er', 'Tm fraction': 'Tm', 'Ta fraction': 'Ta', 'Os fraction': 'Os', 'Pt fraction': 'Pt',
                'Th fraction': 'Th', 'U fraction': 'U', 'Pu fraction': 'Pu','Se fraction': 'Se','I fraction': 'I','Br fraction': 'Br',
                'Sr fraction': 'Sr','Ru fraction': 'Ru','Cl fraction': 'Cl','Te fraction': 'Te','Re fraction': 'Re','Lu fraction': 'Lu',
                'Cs fraction': 'Cs','Tl fraction': 'Tl','Rb fraction': 'Rb','Ba fraction': 'Ba','Ca fraction': 'Ca',
                'W fraction': 'W'})
           # st.write(df_magnetism.columns)     
            with open('ordering_feature_names.pkl', 'rb') as f:
                    expected_feature_names = pickle.load(f)
            
            
            def prepare_data_for_prediction(df):
                processed_df = df.copy()
                categorical_mappings = {
                    'HOMO_element_imputed':["Ag", "Al", "As", "Au", "B", "Ba", "Be", "Bi", "Br", "C", "Ca", "Cd", "Ce", "Cl", "Co", "Cr", "Cs",  
                            "Cu", "Dy", "Er", "Eu", "F", "Fe", "Ga", "Gd", "Ge", "H", "Hf", "Hg", "Ho", "I", "In", "Ir", "K",  
                            "La", "Li", "Lu", "Mg", "Mn", "Mo", "N", "Na", "Nb", "Nd", "Ni", "Np", "O", "Os", "P", "Pa", "Pb",  
                            "Pd", "Pm", "Pr", "Pt", "Pu", "Rb", "Re", "Rh", "Ru", "S", "Sb", "Sc", "Se", "Si", "Sm", "Sn", "Sr",  
                            "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl", "Tm", "U", "V", "W", "Xe", "Y", "Zn", "Zr"
                    ],
                    'LUMO_element_imputed':[ "Ag", "Al", "As", "Au", "B", "Ba", "Be", "Bi", "Br", "C", "Ca", "Cd", "Ce", "Cl", "Co", "Cr", "Cs",
                            "Cu", "Dy", "Er", "Eu", "F", "Fe", "Ga", "Gd", "Ge", "H", "Hf", "Hg", "Ho", "I", "In", "Ir", "K", "La", "Li",
                            "Lu", "Mg", "Mn", "Mo", "N", "Na", "Nb", "Nd", "Ni", "Np", "O", "Os", "P", "Pa", "Pb", "Pd", "Pm", "Pr", "Pt",
                            "Pu", "Rb", "Re", "Rh", "Ru", "S", "Sb", "Sc", "Se", "Si", "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te", "Th", "Ti",
                            "Tl", "Tm", "U", "V", "W", "Xe", "Y", "Zn", "Zr"],
                    'LUMO_character_imputed': ['p', 's','f'],
                    'HOMO_character_imputed':['f','s','p'],
                            
                }
                for column, values in categorical_mappings.items():
                    for value in values:
                        column_name = f"{column}_{value}"
                        processed_df[column_name] = (processed_df[column] == value).astype(int)
                
                    processed_df = processed_df.drop(column, axis=1)

                
                

                processed_df = processed_df.reindex(columns=expected_feature_names, fill_value=0)

                #processed_df
                X = processed_df.copy()

                #pandas2.2.3earlier
                return X
            class StdScaler(BaseEstimator, TransformerMixin):
                def __init__(self, columns=None):
                    self.columns = columns
                    self.scaler = StandardScaler()

                def fit(self, X, y=None):
                    if (self.columns is None):
                        num_cols = [i for i in X.columns if len(np.unique(X[i])) > 5]
                        self.columns = num_cols
                    self.scaler.fit(X[self.columns])
                    return self

                def transform(self, X, y=None):
                    scaled_data = self.scaler.transform(X[self.columns])
                    result_df = pd.DataFrame(scaled_data, columns=self.columns, index=X.index)
                    return pd.concat([X.drop(columns=self.columns), result_df], axis=1)
            # scaler=StdScaler(columns=numerical_cols)
            # df_magnetism=scaler.fit_transform(df_magnetism)
            # df_magnetism
            scaling_pipeline_order=joblib.load("scalingpipe_order.pkl")
            df_mag_order=scaling_pipeline_order.transform(df_mag_order)
            #df_mag_order
            predictions = model_order.predict(prepare_data_for_prediction(df_mag_order))
            st.session_state["mag_order_status"]=predictions
            # Display results
            # st.write("Prediction (Magnetic Ordering: FM/FiM/AFM):", predictions)
            mag_status = st.session_state["mag_status"]
            if(mag_status==0):
                st.markdown("<h1 style='font-size: 18px; color: red;'>Non magnetic</h1>", unsafe_allow_html=True)
                #st.write("Non magnetic")
            else:
                if(predictions==1):
                    st.markdown("<h1 style='font-size: 18px; color: red;'>Magnetic ordering is FM</h1>", unsafe_allow_html=True)
                   # st.write("Magnetic ordering is FM")
                elif(predictions==0):
                    st.markdown("<h1 style='font-size: 18px; color: red;'>Magnetic ordering is AFM</h1>", unsafe_allow_html=True)
                    #st.write("Magnetic ordering is AFM")
                else:
                    st.markdown("<h1 style='font-size: 18px; color: red;'>Magnetic ordering is FiM</h1>", unsafe_allow_html=True)
                    #st.write("Magnetic ordering is FiM")
        
            #0-AFM,1-FM,2-FiM
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please extract features first before predicting.")



# Magnetism ordering temperature prediction
if st.button("Predict Ordering temperature"):
    model_temp= joblib.load("temp_pred_stack_model.pkl")
    if "features_df" in st.session_state:  # Ensure df is available
        try:
            df = st.session_state["features_df"]
            df_temp=df.drop(['composition','oxidation_composition','Pm fraction', 'Tc fraction', 'Ac fraction', 'Pa fraction', 'Xe fraction', 'MagpieData minimum NpValence',
                    'MagpieData minimum NpUnfilled', 'MagpieData minimum GSbandgap',
                    'PymatgenData maximum block','10-norm',
                    '3-norm',
                    '5-norm',
                    '7-norm',
                    'DemlData maximum first_ioniz',
                    'DemlData maximum molar_vol',
                    'DemlData maximum mus_fere',
                    'DemlData mean boiling_point',
                    'DemlData mean first_ioniz',
                    'DemlData mean heat_cap',
                    'DemlData mean mus_fere',
                    'DemlData minimum boiling_point',
                    'DemlData minimum first_ioniz',
                    'DemlData minimum molar_vol',
                    'DemlData minimum mus_fere',
                    'DemlData range boiling_point',
                    'DemlData range electric_pol',
                    'DemlData range first_ioniz',
                    'DemlData range heat_fusion',
                    'DemlData range molar_vol',
                    'DemlData range mus_fere',
                    'DemlData std_dev FERE correction',
                    'DemlData std_dev GGAU_Etot',
                    'DemlData std_dev boiling_point',
                    'DemlData std_dev electric_pol',
                    'DemlData std_dev first_ioniz',
                    'DemlData std_dev heat_cap',
                    'DemlData std_dev heat_fusion',
                    'DemlData std_dev molar_vol',
                    'DemlData std_dev mus_fere',
                    'LUMO_energy',
                    'MEGNetElementData maximum embedding 16',
                    'MEGNetElementData maximum embedding 6',
                    'MEGNetElementData mean embedding 10',
                    'MEGNetElementData mean embedding 11',
                    'MEGNetElementData mean embedding 12',
                    'MEGNetElementData mean embedding 14',
                    'MEGNetElementData mean embedding 15',
                    'MEGNetElementData mean embedding 16',
                    'MEGNetElementData mean embedding 5',
                    'MEGNetElementData mean embedding 6',
                    'MEGNetElementData mean embedding 7',
                    'MEGNetElementData mean embedding 8',
                    'MEGNetElementData mean embedding 9',
                    'MEGNetElementData minimum embedding 10',
                    'MEGNetElementData minimum embedding 11',
                    'MEGNetElementData minimum embedding 7',
                    'MEGNetElementData minimum embedding 8',
                    'MEGNetElementData minimum embedding 9',
                    'MEGNetElementData range embedding 1',
                    'MEGNetElementData range embedding 10',
                    'MEGNetElementData range embedding 16',
                    'MEGNetElementData range embedding 3',
                    'MEGNetElementData range embedding 6',
                    'MEGNetElementData range embedding 8',
                    'MEGNetElementData range embedding 9',
                    'MEGNetElementData std_dev embedding 1',
                    'MEGNetElementData std_dev embedding 10',
                    'MEGNetElementData std_dev embedding 11',
                    'MEGNetElementData std_dev embedding 12',
                    'MEGNetElementData std_dev embedding 13',
                    'MEGNetElementData std_dev embedding 14',
                    'MEGNetElementData std_dev embedding 15',
                    'MEGNetElementData std_dev embedding 16',
                    'MEGNetElementData std_dev embedding 2',
                    'MEGNetElementData std_dev embedding 3',
                    'MEGNetElementData std_dev embedding 4',
                    'MEGNetElementData std_dev embedding 5',
                    'MEGNetElementData std_dev embedding 6',
                    'MEGNetElementData std_dev embedding 7',
                    'MEGNetElementData std_dev embedding 8',
                    'MEGNetElementData std_dev embedding 9',
                    'MagpieData avg_dev AtomicWeight',
                    'MagpieData avg_dev Column',
                    'MagpieData avg_dev CovalentRadius',
                    'MagpieData avg_dev Electronegativity',
                    'MagpieData avg_dev GSbandgap',
                    'MagpieData avg_dev GSmagmom',
                    'MagpieData avg_dev GSvolume_pa',
                    'MagpieData avg_dev MeltingT',
                    'MagpieData avg_dev NUnfilled',
                    'MagpieData avg_dev NValence',
                    'MagpieData avg_dev NfUnfilled',
                    'MagpieData avg_dev NfValence',
                    'MagpieData avg_dev NpUnfilled',
                    'MagpieData avg_dev NpValence',
                    'MagpieData avg_dev NsUnfilled',
                    'MagpieData avg_dev NsValence',
                    'MagpieData avg_dev Row',
                    'MagpieData avg_dev SpaceGroupNumber',
                    'MagpieData maximum AtomicWeight',
                    'MagpieData maximum Column',
                    'MagpieData maximum Electronegativity',
                    'MagpieData maximum NUnfilled',
                    'MagpieData maximum NpValence',
                    'MagpieData maximum NsUnfilled',
                    'MagpieData maximum Row',
                    'MagpieData mean AtomicWeight',
                    'MagpieData mean Column',
                    'MagpieData mean CovalentRadius',
                    'MagpieData mean Electronegativity',
                    'MagpieData mean GSmagmom',
                    'MagpieData mean MeltingT',
                    'MagpieData mean NUnfilled',
                    'MagpieData mean NValence',
                    'MagpieData mean NdValence',
                    'MagpieData mean NfValence',
                    'MagpieData mean NpValence',
                    'MagpieData mean NsUnfilled',
                    'MagpieData mean NsValence',
                    'MagpieData mean Number',
                    'MagpieData mean Row',
                    'MagpieData mean SpaceGroupNumber',
                    'MagpieData minimum AtomicWeight',
                    'MagpieData minimum Column',
                    'MagpieData minimum CovalentRadius',
                    'MagpieData minimum Electronegativity',
                    'MagpieData minimum NsUnfilled',
                    'MagpieData minimum NsValence',
                    'MagpieData minimum Row',
                    'MagpieData minimum SpaceGroupNumber',
                    'MagpieData mode AtomicWeight',
                    'MagpieData mode Column',
                    'MagpieData mode CovalentRadius',
                    'MagpieData mode Electronegativity',
                    'MagpieData mode GSbandgap',
                    'MagpieData mode GSmagmom',
                    'MagpieData mode MeltingT',
                    'MagpieData mode NdValence',
                    'MagpieData mode NpValence',
                    'MagpieData mode NsValence',
                    'MagpieData mode Number',
                    'MagpieData mode Row',
                    'MagpieData mode SpaceGroupNumber',
                    'MagpieData range AtomicWeight',
                    'MagpieData range Column',
                    'MagpieData range CovalentRadius',
                    'MagpieData range Electronegativity',
                    'MagpieData range GSbandgap',
                    'MagpieData range GSmagmom',
                    'MagpieData range GSvolume_pa',
                    'MagpieData range MendeleevNumber',
                    'MagpieData range NUnfilled',
                    'MagpieData range NValence',
                    'MagpieData range NdUnfilled',
                    'MagpieData range NfUnfilled',
                    'MagpieData range NfValence',
                    'MagpieData range NpUnfilled',
                    'MagpieData range NpValence',
                    'MagpieData range NsUnfilled',
                    'MagpieData range NsValence',
                    'MagpieData range Number',
                    'MagpieData range Row',
                    'MagpieData range SpaceGroupNumber',
                    'PymatgenData maximum electrical_resistivity',
                    'PymatgenData maximum group',
                    'PymatgenData mean block',
                    'PymatgenData mean electrical_resistivity',
                    'PymatgenData mean group',
                    'PymatgenData minimum group',
                    'PymatgenData range block',
                    'PymatgenData range bulk_modulus',
                    'PymatgenData range electrical_resistivity',
                    'PymatgenData range group',
                    'PymatgenData range thermal_conductivity',
                    'PymatgenData range velocity_of_sound',
                    'PymatgenData std_dev block',
                    'PymatgenData std_dev bulk_modulus',
                    'PymatgenData std_dev electrical_resistivity',
                    'PymatgenData std_dev group',
                    'PymatgenData std_dev thermal_conductivity',
                    'PymatgenData std_dev velocity_of_sound',
                    'Yang delta',
                    'avg p valence electrons',
                    'frac d valence electrons',
                    'frac f valence electrons',
                    'frac p valence electrons',
                    'mean AtomicRadius',
                    'mean EN difference',
                    'std_dev EN difference','Lu fraction',
                    'MagpieData minimum GSmagmom',
                    'Ir fraction',
                    'Ru fraction',
                    'W fraction',
                    'Th fraction',
                    'Ta fraction',
                    'Mg fraction',
                    'Zr fraction',
                    'Sc fraction',
                    'MagpieData minimum NfUnfilled',
                    'Cl fraction',
                    'Br fraction',
                    'Ag fraction',
                    'Au fraction',
                    'Zn fraction',
                    'Pu fraction',
                    'Os fraction',
                    'Tb fraction',
                    'Bi fraction',
                    'F fraction',
                    'MagpieData mode NsUnfilled',
                    'Hf fraction',
                    'Ho fraction',
                    'Hg fraction',
                    'Na fraction',
                    'K fraction',
                    'Er fraction',
                    'Rb fraction',
                    'Np fraction',
                    'MagpieData maximum NsValence'],axis=1)
            df_temp = df_temp.rename(columns={'HOMO_element': 'HOMO_element_imputed', 'LUMO_element': 'LUMO_element_imputed','LUMO_character': 'LUMO_character_imputed','HOMO_character': 'HOMO_character_imputed'})
            
            # Separate categorical and numerical columns
            categorical_cols = df_temp.select_dtypes(include=['object']).columns
            
            numerical_cols = df_temp.select_dtypes(exclude=['object']).columns
            
            df_temp = df_temp.rename(columns={
                'H fraction': 'H', 'Li fraction': 'Li', 'B fraction': 'B', 'C fraction': 'C', 'N fraction': 'N',
                'O fraction': 'O', 'F fraction': 'F', 'Na fraction': 'Na', 'Mg fraction': 'Mg', 'Al fraction': 'Al',
                'Si fraction': 'Si', 'P fraction': 'P', 'S fraction': 'S', 'Sc fraction': 'Sc', 'Ti fraction': 'Ti',
                'V fraction': 'V', 'Cr fraction': 'Cr', 'Mn fraction': 'Mn', 'Fe fraction': 'Fe', 'Co fraction': 'Co',
                'Ni fraction': 'Ni', 'Cu fraction': 'Cu', 'Zn fraction': 'Zn', 'Ga fraction': 'Ga', 'Ge fraction': 'Ge',
                'Y fraction': 'Y', 'Nb fraction': 'Nb', 'Mo fraction': 'Mo', 'Tc fraction': 'Tc', 'Rh fraction': 'Rh',
                'Pd fraction': 'Pd', 'Ag fraction': 'Ag', 'In fraction': 'In', 'Sn fraction': 'Sn', 'Sb fraction': 'Sb',
                'La fraction': 'La', 'Ce fraction': 'Ce', 'Pr fraction': 'Pr', 'Nd fraction': 'Nd', 'Sm fraction': 'Sm',
                'Eu fraction': 'Eu', 'Gd fraction': 'Gd', 'Tb fraction': 'Tb', 'Dy fraction': 'Dy', 'Ho fraction': 'Ho',
                'Er fraction': 'Er', 'Tm fraction': 'Tm', 'Ta fraction': 'Ta', 'Os fraction': 'Os', 'Pt fraction': 'Pt',
                'Th fraction': 'Th', 'U fraction': 'U', 'Pu fraction': 'Pu','Se fraction': 'Se','I fraction': 'I','Pb fraction': 'Pb',
                'Sr fraction': 'Sr','Te fraction': 'Te','Re fraction': 'Re','Cd fraction': 'Cd','As fraction': 'As',
                'Cs fraction': 'Cs','Tl fraction': 'Tl','Ba fraction': 'Ba','Ca fraction': 'Ca',
                })
           # st.write(df_magnetism.columns)     
            # with open('magnetism_feature_names.pkl', 'rb') as f:
            #         expected_feature_names = pickle.load(f)
            expected_feature_names=['MagpieData avg_dev MendeleevNumber',
                    'MEGNetElementData mean embedding 3',
                    'MEGNetElementData mean embedding 2', 'MagpieData mean NdUnfilled',
                    '2-norm', 'MagpieData avg_dev NdUnfilled', 'Fe',
                    'MagpieData mean GSvolume_pa', 'MagpieData mean NpUnfilled',
                    'DemlData mean molar_vol', 'HOMO_energy',
                    'MEGNetElementData mean embedding 13', 'band center',
                    'avg d valence electrons', 'MagpieData mode MendeleevNumber', 'Co',
                    'MagpieData avg_dev Number', 'PymatgenData mean velocity_of_sound',
                    'LUMO_element_imputed_Fe', 'MagpieData avg_dev NdValence',
                    'PymatgenData mean thermal_conductivity', 'MagpieData maximum GSmagmom',
                    'DemlData mean electric_pol', 'Yang omega',
                    'PymatgenData mean bulk_modulus', 'HOMO_element_imputed_Fe',
                    'MEGNetElementData mean embedding 1', 'MagpieData mode NdUnfilled',
                    'DemlData mean heat_fusion', 'MagpieData minimum MendeleevNumber',
                    'minimum EN difference', 'DemlData maximum heat_fusion',
                    'MEGNetElementData minimum embedding 1',
                    'DemlData maximum electric_pol', 'DemlData range heat_cap',
                    'MEGNetElementData maximum embedding 8', 'MagpieData mean NfUnfilled',
                    'MEGNetElementData range embedding 12', 'MagpieData mean GSbandgap',
                    'MagpieData maximum NdUnfilled', 'DemlData mean GGAU_Etot',
                    'MagpieData maximum GSbandgap', 'maximum EN difference',
                    'MEGNetElementData range embedding 4', 'MagpieData maximum GSvolume_pa',
                    'MagpieData minimum NUnfilled', 'MEGNetElementData range embedding 7',
                    'MEGNetElementData maximum embedding 2',
                    'MEGNetElementData maximum embedding 3',
                    'MEGNetElementData maximum embedding 12',
                    'MEGNetElementData range embedding 2', 'DemlData mean FERE correction',
                    'avg f valence electrons', 'MEGNetElementData mean embedding 4',
                    'MEGNetElementData maximum embedding 1',
                    'MagpieData maximum NpUnfilled', 'Mn', 'frac s valence electrons',
                    'range AtomicRadius', 'MEGNetElementData minimum embedding 15',
                    'DemlData maximum boiling_point', 'MagpieData maximum CovalentRadius',
                    'MagpieData mode GSvolume_pa', 'MEGNetElementData maximum embedding 5',
                    'MEGNetElementData minimum embedding 3', 'avg s valence electrons',
                    'DemlData minimum heat_fusion', 'DemlData maximum heat_cap',
                    'MEGNetElementData maximum embedding 9',
                    'MEGNetElementData minimum embedding 5',
                    'MEGNetElementData range embedding 15', 'O', 'range EN difference',
                    'MEGNetElementData range embedding 11',
                    'DemlData range FERE correction', 'MagpieData maximum MeltingT',
                    'HOMO_character_imputed_s', 'MagpieData minimum MeltingT',
                    'MagpieData maximum MendeleevNumber',
                    'MEGNetElementData maximum embedding 13', 'DemlData minimum GGAU_Etot',
                    'MEGNetElementData minimum embedding 13',
                    'MEGNetElementData minimum embedding 14',
                    'MEGNetElementData range embedding 13', 'MagpieData maximum NdValence',
                    'MagpieData maximum NValence', 'MagpieData minimum NValence',
                    'DemlData maximum GGAU_Etot', 'DemlData range GGAU_Etot',
                    'MagpieData maximum SpaceGroupNumber', 'gap_AO',
                    'MagpieData mode NValence', 'MagpieData maximum Number',
                    'PymatgenData minimum bulk_modulus',
                    'MEGNetElementData minimum embedding 6', 'MagpieData minimum Number',
                    'PymatgenData minimum velocity_of_sound',
                    'PymatgenData minimum thermal_conductivity',
                    'MagpieData maximum NfUnfilled',
                    'MEGNetElementData minimum embedding 4']
            
            def prepare_data_for_prediction(df):
                processed_df = df.copy()
                categorical_mappings = {
                    'HOMO_element_imputed':["Ag", "Al", "As", "Au", "B", "Ba", "Be", "Bi", "Br", "C", "Ca", "Cd", "Ce", "Cl", "Co", "Cr", "Cs",  
                            "Cu", "Dy", "Er", "Eu", "F", "Fe", "Ga", "Gd", "Ge", "H", "Hf", "Hg", "Ho", "I", "In", "Ir", "K",  
                            "La", "Li", "Lu", "Mg", "Mn", "Mo", "N", "Na", "Nb", "Nd", "Ni", "Np", "O", "Os", "P", "Pa", "Pb",  
                            "Pd", "Pm", "Pr", "Pt", "Pu", "Rb", "Re", "Rh", "Ru", "S", "Sb", "Sc", "Se", "Si", "Sm", "Sn", "Sr",  
                            "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl", "Tm", "U", "V", "W", "Xe", "Y", "Zn", "Zr"
                    ],
                    'LUMO_element_imputed':[ "Ag", "Al", "As", "Au", "B", "Ba", "Be", "Bi", "Br", "C", "Ca", "Cd", "Ce", "Cl", "Co", "Cr", "Cs",
                            "Cu", "Dy", "Er", "Eu", "F", "Fe", "Ga", "Gd", "Ge", "H", "Hf", "Hg", "Ho", "I", "In", "Ir", "K", "La", "Li",
                            "Lu", "Mg", "Mn", "Mo", "N", "Na", "Nb", "Nd", "Ni", "Np", "O", "Os", "P", "Pa", "Pb", "Pd", "Pm", "Pr", "Pt",
                            "Pu", "Rb", "Re", "Rh", "Ru", "S", "Sb", "Sc", "Se", "Si", "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te", "Th", "Ti",
                            "Tl", "Tm", "U", "V", "W", "Xe", "Y", "Zn", "Zr"],
                    'LUMO_character_imputed': ['p', 's','f'],
                    'HOMO_character_imputed':['f','s','p'],
                            
                }
                for column, values in categorical_mappings.items():
                    for value in values:
                        column_name = f"{column}_{value}"
                        processed_df[column_name] = (processed_df[column] == value).astype(int)
                
                    processed_df = processed_df.drop(column, axis=1)

                
                

                processed_df = processed_df.reindex(columns=expected_feature_names, fill_value=0)

                #processed_df
                X = processed_df.copy()

                
                return X
            #st.write(prepare_data_for_prediction(df_magnetism).columns.to_list())
            
            
            
            # def compare_features(df, feature_names):
            #     # Get the columns in df_magnetism
            #     df_columns = set(df.columns)
                
            #     # Get the feature names from the model
            #     model_features = set(feature_names)
                
            #     # Find missing features in df_magnetism
            #     missing_features = model_features - df_columns
                
            #     # Find extra features in df_magnetism
            #     extra_features = df_columns - model_features
                
            #     return missing_features, extra_features

            # # Compare features
            # missing_features, extra_features = compare_features(prepare_data_for_prediction(df_magnetism), expected_feature_names)

            # # Print results
            # if missing_features:
            #     st.write("Missing features in df_magnetism:", missing_features)
            # else:
            #     st.write("No missing features in df_magnetism.")

            # if extra_features:
            #     st.write("Extra features in df_magnetism:", extra_features)
            # else:
            #     st.write("No extra features in df_magnetism.")
           
            class StdScaler(BaseEstimator, TransformerMixin):
                def __init__(self, columns=None):
                    self.columns = columns
                    self.scaler = StandardScaler()

                def fit(self, X, y=None):
                    if (self.columns is None):
                        num_cols = [i for i in X.columns if len(np.unique(X[i])) > 5]
                        self.columns = num_cols
                    self.scaler.fit(X[self.columns])
                    return self

                def transform(self, X, y=None):
                    scaled_data = self.scaler.transform(X[self.columns])
                    result_df = pd.DataFrame(scaled_data, columns=self.columns, index=X.index)
                    return pd.concat([X.drop(columns=self.columns), result_df], axis=1)
            # scaler=StdScaler(columns=numerical_cols)
            # df_magnetism=scaler.fit_transform(df_magnetism)
            # df_magnetism
            df_temp.fillna(0, inplace=True)

            scaling_pipeline_temp=joblib.load("scalingpipe_temp.pkl")
            df_temp=scaling_pipeline_temp.transform(df_temp)
            

            predictions = model_temp.predict(prepare_data_for_prediction(df_temp))
            mag_status = st.session_state["mag_status"]
            if(mag_status==0):
                st.markdown("<h1 style='font-size: 18px; color: red;'>No cordering temperature</h1>", unsafe_allow_html=True)
                #st.write("No cordering temperature")
            else:
                mag_order_status = st.session_state["mag_order_status"]
                if(mag_order_status==0):
                    st.markdown("<h1 style='font-size: 18px; color: red;'>Neel temperature:</h1>", unsafe_allow_html=True)
                    st.write(predictions)
                elif(mag_order_status==1):
                    st.markdown("<h1 style='font-size: 18px; color: red;'>Curie temperature:</h1>", unsafe_allow_html=True)
                    st.write(predictions)
                else:
                    st.markdown("<h1 style='font-size: 18px; color: red;'>Curie temperature:</h1>", unsafe_allow_html=True)
                    st.write(predictions)
                
        

            # Display results
            
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please extract features first before predicting.")