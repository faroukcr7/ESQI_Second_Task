from sklearn.base import BaseEstimator, TransformerMixin


class AttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        # Calculate the estimated delivery time and actual delivery time in working days.
        # This would allow us to exclude hollidays that could influence delivery times.
        # If the order_delivered_customer_date is null, it returns 0.
        df['wd_estimated_delivery_time'] = df.apply(lambda x: cal.get_working_days_delta(x.order_aproved_at,
                                                                                         x.order_estimated_delivery_date),
                                                    axis=1)
        df['wd_actual_delivery_time'] = df.apply(lambda x: cal.get_working_days_delta(x.order_aproved_at,
                                                                                      x.order_delivered_customer_date),
                                                 axis=1)

        # Calculate the time between the actual and estimated delivery date. If negative was delivered early, if positive was delivered late.
        df['wd_delivery_time_delta'] = df.wd_actual_delivery_time - df.wd_estimated_delivery_time

        # Calculate the time between the actual and estimated delivery date. If negative was delivered early, if positive was delivered late.
        df['is_late'] = df.order_delivered_customer_date > df.order_estimated_delivery_date

        # Calculate the average product value.
        df['average_product_value'] = df.order_products_value / df.order_items_qty

        # Calculate the total order value
        df['total_order_value'] = df.order_products_value + df.order_freight_value

        # Calculate the order freight ratio.
        df['order_freight_ratio'] = df.order_freight_value / df.order_products_value

        # Calculate the order freight ratio.
        df['purchase_dayofweek'] = df.order_purchase_timestamp.dt.dayofweek

        # With that we can remove the timestamps from the dataset
        cols2drop = ['order_purchase_timestamp', 'order_aproved_at', 'order_estimated_delivery_date',
                     'order_delivered_customer_date']
        df.drop(cols2drop, axis=1, inplace=True)

        return df