from . import NEO_DB

import pandas as pd
import neo4j

if __name__ == "__main__":

    result = NEO_DB.query("""MATCH (art:SDArticle {source:"sdapi"}) RETURN count(art) AS count""")[0]
    print(result)


"""
MATCH (a:SDArticle {source: "sdapi"}) 
WITH collect(a.doi) as L WITH size(L) AS N, L  
WITH L, toInteger(floor(N * 0.009)) AS N_train, toInteger(floor(N * 0.001)) AS N_valid 
WITH L, N_valid, apoc.coll.randomItems(L, N_train, false) AS trainset 
WITH N_valid, apoc.coll.disjunction(L, trainset) AS L2, trainset 
WITH L2, trainset, apoc.coll.randomItems(L2, N_valid, false) AS validation 
WITH apoc.coll.disjunction(L2, validation) AS testset, trainset, validation 
WITH {trainset: trainset, validation: validation, testset: testset} AS dataset 
UNWIND ['train', 'validation', 'test'] AS subset 
UNWIND dataset[subset] AS doi 
MATCH
  (artile:SDArticle {source: "sdapi", doi: doi})-->(figure:SDFigure)-->(panel:SDPanel)-->(tag:SDTag) // For each DOI we get the figures and then their panels
WITH subset, doi, figure,panel
ORDER BY panel.panel_id ASC
  WITH DISTINCT figure, subset, doi, panel, panel.panel_id AS panel_ids
RETURN doi, 
  subset,
  split(figure.href,"=")[1] AS figure_id,
  COLLECT(panel_ids) AS panel_ids,
  apoc.text.join(COLLECT(replace(panel.formatted_caption,"<sd-panel>",'<sd-panel id="'+panel_ids+'">')), "") AS caption
  LIMIT 10
"""


