"""
Copyright 2018 CS Systèmes d'Information

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
from pkgutil import extend_path
from ikats.algo.ts_cut.ts_cut import cut_ts, TsCut, cut_ds_multiprocessing
from ikats.algo.ts_cut.ds_cut_from_metric import cut_ds_from_metric
from ikats.algo.ts_cut.ds_cut import dataset_cut
__path__ = extend_path(__path__, __name__)
