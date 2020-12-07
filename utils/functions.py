import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import os
from datetime import *
from sklearn.cluster import KMeans

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
        return 'Meilleurs clients'
    elif df['rfm_concat_score'] == '111':
        return 'Clients bon marché perdus'    
    elif ((df['f_score'] == 4) and (df['m_score'] <= 3) and (df['r_score'] >= 3)):
        return 'Clients fidèles'
    elif (df['m_score'] > 3):
        return 'Gros dépensiers'
    elif (df['r_score'] <= 2):
        return 'Presque perdus'
    elif ((df['r_score'] == 1) and (df['f_score'] >= 1) and (df['m_score'] >= 1)):
        return 'Clients perdus'
    else:
        return 'Autres clients'
   
    
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

def plotKmeansClustersWcss(data):
    wcss = []
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, init="k-means++")
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(12, 6))
    plt.grid()
    plt.plot(range(1, 21), wcss, linewidth=2, color="red", marker ="8")
    plt.xlabel("nombre des clusters")
    plt.xticks(np.arange(1, 21, 1))
    plt.ylabel("Somme moyenne des erreurs quadratiques")
    plt.title('Clustering avec critère de coude')
    plt.show()
    
def plotKmeansClusters(clusters_sample, X_tsne):
    plt.figure(figsize=(15, 12))
    plt.axis([np.min(X_tsne[:, 0] * 1.1), np.max(X_tsne[:, 0] * 1.1),
              np.min(X_tsne[:, 1] * 1.1), np.max(X_tsne[:, 1] * 1.1)])

    colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid',
              'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan',
              'navy', 'gray', 'purple']

    for i in range(len(np.unique(clusters_sample))):
        idx = clusters_sample == i
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=i, s=10, c=colors[i])
        plt.title('T-SNE via k-means avec les variables de RFM')

    plt.legend(loc='upper left', markerscale=2)
    plt.show()

# Fonction pour générer les données glissantes
def createFeaturesCustomers(dataCustomer, startDate, endDate):
    maxDuration = 20 - int(startDate.split()[1])
    minDuration = 20 - int(endDate.split()[1])
    dataframeCustomer = \
        dataCustomer[(dataCustomer['duration_frequence_order']
                      <= maxDuration)
                     & (dataCustomer['duration_frequence_order']
                        >= minDuration)]

    return dataframeCustomer

# Fonction pour générer les données glissantes V2
def createFeaturesCustomersV2(AllData, startDate, endDate):

    AllData = \
    AllData[(AllData['order_purchase_timestamp']>=startDate)
          & (AllData['order_purchase_timestamp']<=endDate)]
    
    
    customers_copy = \
        pd.read_csv('data/olist_customers_dataset.csv', sep=',', engine='python')
    
    # calcul de la durée de dernier achat :
    AllDataRecenceByCustomer = \
        AllData.groupby('customer_id')['order_purchase_timestamp']\
        .max().reset_index(name='Last_order_date')
    customers_Behaviour = pd.merge(customers_copy, AllDataRecenceByCustomer,
                                   left_on='customer_id',
                                   right_on='customer_id',
                                   how='inner')

    customers_Behaviour['max_date_order'] =\
        AllData['order_purchase_timestamp'].max()
    # Formater les dates pour faire les calculs
    customers_Behaviour.max_date_order = customers_Behaviour.max_date_order\
        .apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    customers_Behaviour.Last_order_date = customers_Behaviour.Last_order_date\
        .apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    # calcul de la recence (en jour) :
    customers_Behaviour['recency_order'] =\
        (customers_Behaviour.max_date_order-customers_Behaviour.Last_order_date)
    customers_Behaviour.recency_order = \
        abs(customers_Behaviour.recency_order
            .apply(lambda x: x.total_seconds()/86400)).astype(int)    
    
    # calcul de la fréquence d'achat :
    AllDataFrequenceByCustomer =\
        AllData.groupby('customer_id').size().reset_index(name='frequency_order')
    customers_Behaviour =\
        pd.merge(customers_Behaviour, AllDataFrequenceByCustomer,
                 left_on='customer_id', right_on='customer_id', how='inner')
    # customers_Behaviour.shape    
    
    
    # calcul de la dépense du client :
    AllDataAmountByCustomer =\
        AllData.groupby('customer_id')['Total payment_value_Order']\
        .sum().reset_index(name='monetary_amount_order')
    customers_Behaviour =\
        pd.merge(customers_Behaviour, AllDataAmountByCustomer,
                 left_on='customer_id', right_on='customer_id', how='inner')

    # calcul de nombre de mois de fréquence(durée arrondie en mois) :
    FirstOrderByCustomer =\
        AllData.groupby('customer_id')['order_purchase_timestamp']\
        .min().reset_index(name='First_order_date')
    customers_Behaviour =\
        pd.merge(customers_Behaviour, FirstOrderByCustomer,
                 left_on='customer_id', right_on='customer_id', how='inner')
    customers_Behaviour.First_order_date = customers_Behaviour.First_order_date.\
        apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    # calcul de nombre de mois de fréquence(durée arrondie en mois) :
    customers_Behaviour['duration_frequence_order'] =\
        (customers_Behaviour.max_date_order-customers_Behaviour.First_order_date)
    customers_Behaviour.duration_frequence_order =\
        abs(customers_Behaviour.duration_frequence_order
            .apply(lambda x: x.total_seconds()/(86400*30))).astype(int)

    customers_Behaviour.drop(columns=['Last_order_date',
                                      'max_date_order',
                                      'customer_zip_code_prefix'],
                             inplace=True)    
    
    # calcul de panier moyen du client :
    customers_Behaviour['mean_value_order'] =\
        customers_Behaviour['monetary_amount_order']/customers_Behaviour['frequency_order']   
    
    # Calcul des moyennes de satisfactions du client
    # sur l'ensemble de ses commandes :
    AllDataReviewScoreByCustomer =\
        AllData.groupby('customer_id')['review_score'].mean()\
        .reset_index(name='review_score_order')
    customers_Behaviour = pd.merge(customers_Behaviour,
                                   AllDataReviewScoreByCustomer,
                                   left_on='customer_id',
                                   right_on='customer_id',
                                   how='inner')    
 
    # calcul des photos des produits achetés :
    AllDataPhotosProductByCustomer =\
        AllData.groupby('customer_id')['product_photos_qty'].sum()\
        .reset_index(name='sum_product_photos_order')
    customers_Behaviour =\
        pd.merge(customers_Behaviour, AllDataPhotosProductByCustomer,
                 left_on='customer_id', right_on='customer_id', how='inner')
    # calcul de nombre de produits des produits achetés :
    AllDataNbrProductsByCustomer =\
        AllData.groupby('customer_id')['product_id'].size()\
        .reset_index(name='nbr_products_customer')
    customers_Behaviour =\
        pd.merge(customers_Behaviour, AllDataNbrProductsByCustomer,
                 left_on='customer_id', right_on='customer_id', how='inner')

    # calcul de nombre des photos moyen par produit pour un consommateur :
    customers_Behaviour['mean_number_photos'] = abs(
        customers_Behaviour['sum_product_photos_order']
        /
        customers_Behaviour['nbr_products_customer']).astype(int)
    customers_Behaviour.drop(columns=['sum_product_photos_order',
                                      'nbr_products_customer'], inplace=True)
    
    # Calcul de nombre de type de paiement differents :
    AllDataPaymentsTypeByCustomer =\
        AllData.groupby(['customer_id',
                         'order_id'])['nbrPaymentType']\
        .max().reset_index(name='nbr_payments_type')
    customers_Behaviour = pd.merge(customers_Behaviour,
                                   AllDataPaymentsTypeByCustomer,
                                   left_on='customer_id',
                                   right_on='customer_id', how='inner')
    # Suppression des colonnes non utilisées
    customers_Behaviour.drop(columns=['customer_unique_id',
                                      'customer_city',
                                      'order_id',
                                      'customer_state',
                                      'First_order_date'
                                     ], inplace=True)
    
    # Normalisation et standardisation des données :
    customers_Behaviour_Log = customers_Behaviour.copy()
    listVarDensite = ['recency_order', 'frequency_order',
                      'monetary_amount_order',
                      'duration_frequence_order', 'mean_value_order',
                      'review_score_order', 'mean_number_photos',
                      'nbr_payments_type']
    for j, val in enumerate(listVarDensite):
        # transformer les données pour le rendre normales
        # Les variables qui ont un coeeficient de skewness > 2
        if (customers_Behaviour_Log[val].skew() > 2):
            customers_Behaviour_Log[val] =\
                np.log(customers_Behaviour_Log[val] + 1)
        else:
            customers_Behaviour_Log[val] =\
                customers_Behaviour_Log[val]
        jmax = j
    
    return customers_Behaviour_Log