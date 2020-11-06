import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import os
from datetime import *

def _scale_data(data, ranges):
    (x1, x2) = ranges[0]
    d = data[0]
    res = [(d-y1) / (y2-y1) * (x2-x1) + x1
           for d, (y1, y2) in zip(data, ranges)]
    return res


class RadarChart():
    def __init__(self, fig, location, sizes, variables, ranges,
                 n_ordinate_levels=6):

        angles = np.arange(0, 360, 360./len(variables))

        ix, iy = location[:]
        size_x, size_y = sizes[:]

        axes = [fig.add_axes([ix, iy, size_x, size_y],
                             polar=True,
                             frameon=False,
                             label="axes{}".format(i))
                for i in range(len(variables))]

        _, text = axes[0].set_thetagrids(angles, labels=variables, size=13)

        for txt, angle in zip(text, angles):
            if angle > -1 and angle < 181:
                txt.set_rotation(angle - 90)
            else:
                txt.set_rotation(angle - 270)

        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.grid(False)

        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], num=n_ordinate_levels)
            grid_label = [""]+["{:.0f}".format(x) for x in grid[1:-1]]
            ax.set_rgrids(grid, labels=grid_label, angle=angles[i], size=12)
            ax.set_ylim(*ranges[i])

        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def legend(self, *args, **kw):
        self.ax.legend(*args, **kw)

    def title(self, title, *args, **kw):
        self.ax.text(0.9, 1, title, transform=self.ax.transAxes, *args, **kw)

def radar_plot(df, group='cluster', xsize=0.1, ysize=0.05, figsize=(16, 6)):
    """
    Input : Dataframe, colonne des clusters
    Output : Affichage des clusters en radar chart
    """

    # Moyenne des variables par cluster
    df_agg = df.groupby(group)[df.columns].mean()

    # Taille des clusters en %
    clusters_size = round(df[group].value_counts()
                          / df[group].count() * 100, 2)

    # Liste des variables descriptives des clusters
    var_descriptives = [col for col in df.columns if col not in group]

    # Min,max des variables descriptives
    var_descript_ranges = [list(df_agg.describe().loc[['min', 'max'], var])
                           for var in var_descriptives]

    # Nombre de clusters
    n_clusters = df[group].nunique()

    # Liste des index des variables descriptives
    vars_index = list(range(len(var_descriptives)))

    # Dimension de la grille
    n_cols = 3
    m_rows = n_clusters // n_cols

    size_x, size_y = (1 / n_cols), (1 / m_rows)

    fig = plt.figure(figsize=figsize)

    df_agg_index = df_agg.index

    for i_cluster in range(n_clusters):

        ix = i_cluster % n_cols
        iy = m_rows - i_cluster // n_cols
        pos_x = ix * (size_x + xsize)
        pos_y = iy * (size_y + ysize)
        location = [pos_x, pos_y]
        sizes = [size_x, size_y]
        data_plot = np.array(df_agg.loc[df_agg_index[i_cluster],
                                        var_descriptives])
        radar = RadarChart(fig, location, sizes, var_descriptives,
                           var_descript_ranges)
        radar.plot(data_plot, color='b', linewidth=2.0)
        radar.fill(data_plot, alpha=0.2, color='b')

        # Titre du radarchart
        cluster_num = df_agg_index[i_cluster]
        cluster_size = clusters_size[i_cluster]
        radar.title(title=f'cluster n°{cluster_num}\nsize={cluster_size}%',
                    color='r',
                    size=14)
        i_cluster += 1


def getNbDays(start, end, datetimeFormat):
    try :
        diffDays = (datetime.strptime(end, datetimeFormat) - datetime.strptime(start, datetimeFormat)).days
        return diffDays
    except ValueError:
        return -1

def setCategoriesData():
    product_categories_dict = {
        'construction_tools_construction': 'construction',
        'construction_tools_lights': 'construction',
        'construction_tools_safety': 'construction',
        'costruction_tools_garden': 'construction',
        'costruction_tools_tools': 'construction',
        'garden_tools': 'construction',
        'home_construction': 'construction',

        'fashio_female_clothing': 'fashion',
        'fashion_bags_accessories': 'fashion',
        'fashion_childrens_clothes': 'fashion',
        'fashion_male_clothing': 'fashion',
        'fashion_shoes': 'fashion',
        'fashion_sport': 'fashion',
        'fashion_underwear_beach': 'fashion',

        'furniture_bedroom': 'furniture',
        'furniture_decor': 'furniture',
        'furniture_living_room': 'furniture',
        'furniture_mattress_and_upholstery': 'furniture',
        'bed_bath_table': 'furniture',
        'kitchen_dining_laundry_garden_furniture': 'furniture',
        'office_furniture': 'furniture',

        'home_appliances': 'home',
        'home_appliances_2': 'home',
        'home_comfort_2': 'home',
        'home_confort': 'home',
        'air_conditioning': 'home',
        'housewares': 'home',
        'art': 'home',
        'arts_and_craftmanship': 'home',
        'flowers': 'home',
        'cool_stuff': 'home',

        'drinks': 'food_drink',
        'food': 'food_drink',
        'food_drink': 'food_drink',
        'la_cuisine': 'food_drink',

        'electronics': 'electronics',
        'audio': 'electronics',
        'tablets_printing_image': 'electronics',
        'telephony': 'electronics',
        'fixed_telephony': 'electronics',
        'small_appliances': 'electronics',
        'small_appliances_home_oven_and_coffee': 'electronics',
        'computers_accessories': 'electronics',
        'computers': 'electronics',
        'sports_leisure': 'sports_leisure',
        'consoles_games': 'sports_leisure',
        'musical_instruments': 'sports_leisure',
        'toys': 'sports_leisure',
        'cine_photo': 'sports_leisure',
        'dvds_blu_ray': 'sports_leisure',
        'cds_dvds_musicals': 'sports_leisure',
        'music': 'sports_leisure',
        'books_general_interest': 'sports_leisure',
        'books_imported': 'sports_leisure',
        'books_technical': 'sports_leisure',

        'health_beauty': 'health_beauty',
        'perfumery': 'health_beauty',
        'diapers_and_hygiene': 'health_beauty',
        'baby': 'health_beauty',

        'christmas_supplies': 'supplies',
        'stationery': 'supplies',
        'party_supplies': 'supplies',
        'auto': 'supplies',
        'luggage_accessories': 'supplies',

        'watches_gifts': 'gifts',

        'agro_industry_and_commerce': 'misc',
        'industry_commerce_and_business': 'misc',
        'security_and_services': 'misc',
        'signaling_and_security': 'misc',
        'market_place': 'misc',
        'pet_shop': 'misc',
    }
    dfCaregories = pd.DataFrame(list(product_categories_dict.items()), columns=['category', 'product_category_group'])
    return dfCaregories

def getCategoriesGroup(cat_dict,cat_name):
    cat_group = cat_dict[cat_name]

    return cat_group

def rfm_level_type1(df):
    if df['rfm_concat_score'] == '444':
        return 'Best Customers'
    elif ((df['f_score'] == 4) and (df['m_score'] != 4)):
        return 'Loyal Customers'
    elif (df['m_score'] == 4):
        return 'Big Spenders'
    elif (df['r_score'] <= 2):
        return 'Almost Lost'
    elif ((df['r_score'] == 1) and (df['f_score'] >= 1) and (df['m_score'] >= 1)):
        return 'Lost Customers'
    elif df['rfm_concat_score'] == '111':
        return 'Lost Cheap Customers'    
    else:
        return 'Normal customers'
    
   
    
def rfm_level_type2(df):
    if df['rfm_score'] >= 9:
        return '9 +'
    elif ((df['rfm_score'] >= 8) and (df['rfm_score'] < 9)):
        return '8'
    elif ((df['rfm_score'] >= 7) and (df['rfm_score'] < 8)):
        return '7'
    elif ((df['rfm_score'] >= 6) and (df['rfm_score'] < 7)):
        return '6'
    elif ((df['rfm_score'] >= 5) and (df['rfm_score'] < 6)):
        return '5'
    elif ((df['rfm_score'] >= 4) and (df['rfm_score'] < 5)):
        return '4'
    else:
        return '3 -'