

import pandas as pd

datos = pd.read_excel("/content/business.retailsales.xlsx")

datos.head()

datos.dtypes

datos.TipoProducto

datos.describe()

datos.hist(bins=20, figsize=(20,20))

datos.groupby('TipoProducto')['CantidadNeta'].sum()

datos.TipoProducto.replace('Accessories','Art & Sculpture','Basket','Christmas','Easter','Fair Trade Gifts','Furniture','Gift Baskets','Home Decor','Jewelry','Kids','Kitchen','Music','One-of-a-Kind','Recycled Art','Skin Care','Soapstone','Textiles'), (17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), inplace = True

datos.groupby('TipoProducto')['VentaNetaTotal'].sum()

datos.groupby('TipoProducto')['VentaNetaTotal'].mean()

datos.groupby('TipoProducto')['Descuentos'].sum()

datos.groupby('TipoProducto')['Descuentos'].mean()

datos.groupby('TipoProducto')['Devoluciones'].sum()

datos.groupby('TipoProducto')['Devoluciones'].mean()

"""Conclusiónes: El negocio depende de los descuentos para aumentar sus ventas. La categoría de producto más vendida es "Arte y escultura" y "Basket" ya que son los productos que más hay (1427 y 1461), para aumentar más sus ventas de debe realizar descuentos.

Los productos con más demanda son 'Joyería' y 'Cocina' pero los descuentos de estos productos son bajos, se debe aumentar el descuento para aumentar las ventas.

Los siguientes más demandados son Navidad y decoración del hogar

Opciones: Reducir la cantidad de descuento de los productos más vendidos, a causa de esto bajaran las ventas pero aumenta la rentabilidad, la otra opción es seguir manteniendo el descuento en los productos más vendidos.
-Aumentar el descuento en 'Cocina' para aumentar sus ventas.


Conclusions: The business dependent on discounts to increase sales. The best-selling product category is "Art and sculpture" and "Basket" since they are the most popular products (1427 and 1461), to further increase your sales you must make discounts.

The products with the most demand are 'Jewelry' and 'Kitchen' but the discounts for these products are low, the discount must be increased to increase sales.

The next most demanded are Christmas and home decoration

Options: Reduce the amount of discount on the best-selling products, because of this, sales will decrease but profitability increases, the other option is to continue maintaining the discount on the best-selling products. -Increase the discount in 'Kitchen' to increase your sales


"""
