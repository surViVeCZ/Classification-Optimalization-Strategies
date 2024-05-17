# Baby's first DataFrame transformer function

A step by step guide to writing a DataFrame transformer function for our loader. We use these functions to transform the data we load from the database into a format that is more useful for training. Each function takes a DataFrame as input and returns a DataFrame as output. The functions are chained together to transform the data in a series of steps. Every transformer or a group of related transformers is defined in a file in the `transformers` directory. To register a transformer, import it to the `transformers/__init__.py` script as instructed in the file. Every member of the `transformers` module that is callable and starts with `transform_` is automatically run by the loader.

## Step 1: Consider the data
As we use a projection to only load the minimum amount of data from the database, we need to make sure that the data we need is available in the DataFrame. If it is not, we need to add it to the projection. The projection is defined in the `projection.py` script. The projection tells Mongo which fields we want to load from the collection documents.

Take a look around the projection definition. If you need to load a field that is not currently projected, add it to the projection. The file explains how to do this. Next, you'll need to also modify the schema definition in the `schema.py` script. The schema defines the types of the fields in the projection. The schema is used to validate the data we load from the database. If you need to add a field to the projection, you'll need to add it to the schema as well. As with projection, the file explains how to do this.

As a side note, if you changed the projection or schema, make sure the `Config.CACHE` option is disabled before you run your pipeline at the end. The cache will not be invalidated if you change the projection or schema. This will cause the loader to load the old data from the cache and not the new data from the database.

## Step 2: Consider the order
The order of the transformers is important. The data is transformed in the order the transformers are defined in the `transformers/__init__.py` script. Add your transformer registration to the position in the list where you want it to run. Consider the state of the DataFrame after each step. For example, the flatten transform will flatten some embedded documents into the top level of the DataFrame and drop the original fields. If you want to use the original fields, add your transformer before the flatten transform.

## Step 3: Write the function
Your function will take the DataFrame as it was output by the preceding transformers and it will output a DataFrame to be used by the following transformers. With this in mind, take care not to modify fields that do not concern your transformer. If you need to add a new field, make sure you do not overwrite an existing field. If you need to drop a field, make sure you do not drop a field that is used by a following transformer.

Always make sure you apply your transformations to every row in the DataFrame. You can do this by using the `apply` method on the DataFrame. The `apply` method takes a function as input and applies it to every row in the DataFrame. The function you pass to `apply` should take a single argument, the row, and return a modified row. The `apply` method will return a new DataFrame with the modified rows. You can then return this new DataFrame from your transformer function.

There are of course other ways to do this, it depends on what you want to do to the DataFrame. Just remember you're working with a table of data, so you need to work with every row and, for example, not read data you then transform from a single row and write it to all the other rows. Play around with pandas a bit to get a feel for how it works.

## Step 4: Register the function
Simple enough. Import your transformer function to the `transformers/__init__.py` script. The file explains how to do this. The function will now be called by the loader.

## Step 5: Run the loader
Now that you've written your transformer function, you can run the loader to test it. The collections to load and their labels are defined in `Config.COLLECTIONS`. The loader will load the data from the database (remember to check the `CACHE` option), transform it, and save it to the `floor` directory (ha ha) as a parquet. If you (then) don't need to change the projection or schema, you can run the loader with the `CACHE` option enabled. This will load the data from the cache instead of the database. This is much faster.

If you want to see the resulting DataFrame, print it before you return it from your transformer or print the final DataFrame after all the transformers have run. You can also use the `head` method on the DataFrame to print the first few rows. Or load the parquet file in a notebook and play around with it. Use the `pyarrow.parquet` module to load the parquet file. You can then use the `to_pandas` method on the resulting `pyarrow.Table` to get a pandas DataFrame.