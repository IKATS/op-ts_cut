# Operator op-ts_cut

Cutting involves selecting a smaller section of a timeseries or dataset. Possible uses include:

- making a timeseries more manageable when prototyping a model
- restricting points in a time series to a chosen time period
- removing outliers that are too high or too low
- selecting points that fulfil criteria on another column
- selecting an interesting pattern from a time series to search for in other time series

Their are three ways to cut a TS in IKATS:

## Cut DS by metric

### Input and parameters

This operator only takes one input of the functional type **ds_name**.

It also takes 4 inputs from the user:

- **Metric (M)**: the unique metric used as reference to cut the dataset
- **Cut condition**: the python expression determining the retention conditions over the selected metric (M). Example: (M>5) and (M<10) and (M not in \[7,8\])
- **Group by**: the optional criteria used to group timeseries (sharing the same value). Leave blank to not group.
- **FuncId pattern**: Python pattern determining format of output functional identifiers. Example: {fid}_{M}_cut where '{fid}' is the Functional Identifier of the original time series and '{M}' is metric.

### Outputs

The operator has one output of the functional type **ts_list**, which is the list of cut TS (in the same order as input)

## Cut Dataset

Performs a cut on a whole DATASET along time axis: keep only the points situated in specified time interval defined by start/end date or by start date/number of points.
The algorithm has the choice to use a spark implementation or a multiprocessing one.
The choice of spark is done when at least one of the following condition is met :
- at least one time series of the dataset has a size bigger than two times the number of points by chunk of data used by spark (number of points by chunk actual default value : 50000)
- the dataset total number of time series exceeds 100 timeseries

Using spark, cutting by start date/number of points takes more or less 30% more time than cutting by start/end date (because we have to gather size information among the whole range of data for cutting)

Special behaviours:
- When no point is found in specified range for one or several time series, no result is produced for these series, so a dataset of n time series can produced a list of time series whose size is lesser than n.
- when cut number of points exceeds the time series size, the cut does not raise an error but returns points from start cutting date to the time series end date
- when cutting an empty or unknown dataset, an empty result is returned

### Input and parameters

This operator only takes one input of the functional type **ds_name**.

It also takes 3 inputs from the user:

- **Start date**: Start cutting date (millisec)
- **End date**: End cutting date (millisec): optional: if missing, number of points shall be provided instead (mutually exclusive)
- **Number of points**: Number of points retained by the time cut. optional: if missing, end cutting date shall be provided instead (mutually exclusive)

**End date** and **Number of points** are mutually exclusive: exactly one of both must be filled

### Outputs

The operator has one output of the functional type **ts_list**, which is the list of cut TS (in the same order as input)

## Cut TS

Performs a cut on a single time series along time axis.
**Used only in viztools as a visual cut (not accessible via operators list)**

### Input and parameters

This operator only takes one input of the functional type **tsuid**.

It also takes 3 inputs from the user:

- **Start date**: Start cutting date (millisec): optional: if missing, ikats_start_date metadata is used as default value
- **End date**: End cutting date (millisec): optional: if missing, ikats_end_date metadata is used as default value
- **Number of points**: Optional: maximum number of points retained by the time cut.

### Outputs

The operator has one output of the functional type **ts_list**, which is the list of the only cut TS
