import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


def load_data () :
    data_customers = pd.read_csv("../../olist_customers_dataset.csv")
    data_reviews  = pd.read_csv("../../olist_order_reviews_dataset.csv")
    data_orders = pd.read_csv("../../olist_orders_dataset.csv")
    data_needed = pd.concat([data_customers , data_orders ,data_reviews ] , axis=0 )
    data_needed =data_needed.replace(np.nan, 0)
    data_needed = data_needed.drop(columns=['customer_city', 'customer_id','customer_state','customer_unique_id','order_delivered_customer_date',
                                            'review_id' , 'review_creation_date' ,'review_answer_timestamp','review_comment_message','review_comment_title' ,
                                            'order_id','order_status','order_purchase_timestamp','order_estimated_delivery_date','order_delivered_carrier_date','order_approved_at'])
    data_needed =data_needed.replace(np.nan ,0)

    print(data_needed)
    data_needed.to_csv(index=True,path_or_buf="./data_needed.csv")
    return data_needed


def load_modal (data):
    X = data
    #plt.scatter(data['review_score'], data['customer_zip_code_prefix'])
    #plt.title('Data Scatter ')
    #plt.xlabel('customer_zip_code_prefix')
    #plt.ylabel('review_score')
    #plt.show()

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()



    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X)
    plt.scatter(data['review_score'], data['customer_zip_code_prefix'])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()



if __name__ == '__main__':
    data_needed =  load_data()
    load_modal(data_needed)






