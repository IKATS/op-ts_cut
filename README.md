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

Performs a cut on a whole DATASET along time axis: keep only the points situated in specified time interval

#Review#494: Explain how the spark/multiprocessing choice is made, the behavior of the cut for main edge usages (such as 'no points in range' for example)

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
