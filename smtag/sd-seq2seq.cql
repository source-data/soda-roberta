MATCH
  (a:SDArticle {source: "sdapi"})-->(f:SDFigure)-->(p:SDPanel)
WITH p
MATCH (p)-->(observation:CondTag {role: "assayed"})
WITH p, observation
MATCH (p)-->(intervention:CondTag {role: "intervention"})
WITH p, observation, intervention
MATCH (p)-->(method:CondTag {category: "assay"})
WITH DISTINCT
  p,
  COLLECT(DISTINCT observation.type+":"+observation.text) AS observations,
  COLLECT(DISTINCT intervention.type+":"+intervention.text) AS interventions,
  COLLECT(DISTINCT method.text) AS methods
WITH
  id(p) as p_id,
  apoc.text.replace(p.caption, "<[^>]*>", "") as p_caption,
  apoc.coll.sort(interventions) AS interventions,
  apoc.coll.sort(observations) AS observations,
  apoc.coll.sort(methods) AS methods
ORDER BY p_id
WITH
  p_id, p_caption,
  interventions, observations, methods,
  "Causal hypothesis: " + apoc.text.join(interventions, " or ") + " was tested for its influence on " + apoc.text.join(observations, " or ") + " by " + apoc.text.join(methods, ", ") AS serialized
WITH DISTINCT
  COLLECT(toString(p_id)) AS p_ids, 
  serialized,
  COLLECT(p_caption) AS p_captions
UNWIND p_captions as caption
RETURN
  caption + "###tt9HHSlkWoUM###" + serialized + " (sources: " + apoc.text.join(p_ids, ", ") + ")"