MATCH (a:SDArticle {source: "sdapi"}) 
WITH collect(a.doi) as L WITH size(L) AS N, L  
WITH L, toInteger(floor(N * 0.8)) AS N_train, toInteger(floor(N * 0.1)) AS N_valid 
WITH L, N_valid, apoc.coll.randomItems(L, N_train, false) AS trainset 
WITH N_valid, apoc.coll.disjunction(L, trainset) AS L2, trainset 
WITH L2, trainset, apoc.coll.randomItems(L2, N_valid, false) AS validation 
WITH apoc.coll.disjunction(L2, validation) AS testset, trainset, validation 
WITH {trainset: trainset, validation: validation, testset: testset} AS dataset 
UNWIND ['trainset', 'validation', 'testset'] AS subset 
UNWIND dataset[subset] AS doi 
MATCH
  (a:SDArticle {source: "sdapi", doi: doi})-->(f:SDFigure)-->(p:SDPanel) // For each DOI we get the figures and then their panels
WITH p, subset
MATCH
    (p)-->(observation:CondTag {role: "assayed"})<--(:SDTag {in_caption: true}),
    (p)-->(intervention:CondTag {role: "intervention"})<--(:SDTag {in_caption: true}),
    (p)-->(method:CondTag {category: "assay"})<--(:SDTag {in_caption: true})
WITH DISTINCT
  p, subset,
  // use geneprod instead of gene or protein
  apoc.text.replace(observation.type, "gene|protein", "geneprod") as obs_type,
  observation.text AS obs_text,
  apoc.text.replace(intervention.type, "gene|protein", "geneprod") as interv_type,
  intervention.text AS interv_text,
  method.text AS method_text
WITH DISTINCT
  p, subset,
  COLLECT(DISTINCT "*"+obs_type+" "+obs_text+" "+obs_type+"*") AS observations, // This piece of code will generate the tags to be shown in the output
  COLLECT(DISTINCT "*"+interv_type+" "+interv_text+" "+interv_type+"*") AS interventions,
  COLLECT(DISTINCT "*method "+method_text+" method*") AS methods

WITH DISTINCT
  id(p) as p_id, subset,
  // remove HTML tags
  apoc.text.replace(p.caption, "<[^>]*>", "") as p_caption,
  apoc.coll.sort(interventions) AS interventions,
  apoc.coll.sort(observations) AS observations,
  apoc.coll.sort(methods) AS methods

ORDER BY p_id
WITH
  p_id, p_caption, subset,
  interventions, observations, methods,
  apoc.text.join(interventions, " or ") + " was tested for its influence on " + apoc.text.join(observations, " or ") + " by " + apoc.text.join(methods, ", ") AS serialized

WITH DISTINCT
  subset,
  COLLECT(toString(p_id)) AS p_ids, 
  serialized,
  COLLECT(p_caption) AS p_captions
UNWIND p_captions as caption
RETURN
  subset,
  "Tag entities, find causal hypothesis, and experimental method: " + caption + " ###tt9HHSlkWoUM### " + serialized + " [END]" // + " (sources: " + apoc.text.join(p_ids, ", ") + ")"
