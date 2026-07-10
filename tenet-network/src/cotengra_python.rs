//! External path planning through the installed Python `cotengra` package.
//!
//! This module deliberately keeps the boundary narrow: TeNeT lowers
//! [`NetworkIR`] to `inputs/output/size_dict`, Python returns a recycled
//! active-pair path, and the normal Rust [`ContractionPlan`] validation and
//! executor do the rest.

use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::{json, Value};
use tenet::plancache::{
    CotengraMinimize, CotengraPythonConfig, CotengraPythonMethod, CotengraSlicingConfig,
};

use crate::cost::DenseCostModel;
use crate::error::{ContractError, Result};
use crate::ir::NetworkIR;
use crate::optimizer::{ContractionStep, DenseContractionOptimizer};
use crate::plan::{dense_steps_from_active_pair_path, ActivePair, ContractionPlan};
use crate::slice::{slice_plan_for_ordered, SliceKind, SlicedPlan};
use crate::TemporaryLabel;

const PYTHON_PLANNER: &str = r#"
import json
import os
import sys
import traceback

def main():
    spec = json.load(sys.stdin)
    import cotengra as ctg

    inputs = spec["inputs"]
    output = spec["output"]
    size_dict = spec["size_dict"]
    config = spec["config"]
    method = config["method"]
    minimize = config["minimize"]
    max_repeats = config["max_repeats"]
    seed = config["seed"]
    parallel = config["parallel"]

    if method == "auto":
        optimize = "auto"
    elif method == "auto-hq":
        optimize = "auto-hq"
    elif method == "greedy":
        optimize = ctg.GreedyOptimizer()
    elif method == "optimal":
        optimize = ctg.OptimalOptimizer(minimize=minimize)
    elif method == "random-greedy":
        optimize = ctg.RandomGreedyOptimizer(
            max_repeats=max_repeats,
            seed=seed,
            parallel=parallel,
        )
    elif method == "hyper":
        optimize = ctg.HyperOptimizer(
            minimize=minimize,
            max_repeats=max_repeats,
            parallel=parallel,
            progbar=False,
            on_trial_error="raise",
            simulated_annealing_opts=None,
            slicing_opts=None,
            slicing_reconf_opts=None,
            reconf_opts=None,
        )
    else:
        raise ValueError(f"unknown cotengra method: {method}")

    tree = ctg.array_contract_tree(
        inputs,
        output=output,
        size_dict=size_dict,
        optimize=optimize,
        canonicalize=False,
        sort_contraction_indices=False,
    )

    slicing = config["slicing"]
    kind = slicing["kind"]
    if kind == "none":
        pass
    elif kind == "slice":
        tree = tree.slice(
            target_size=slicing["target_size"],
            max_repeats=slicing["max_repeats"],
            allow_outer=slicing["allow_outer"],
            minimize=minimize,
            seed=seed,
        )
    elif kind == "reconfigure":
        reconf_opts = {"forested": slicing["forested"]}
        tree = tree.slice_and_reconfigure(
            target_size=slicing["target_size"],
            step_size=slicing["step_size"],
            max_repeats=slicing["max_repeats"],
            allow_outer=slicing["allow_outer"],
            minimize=minimize,
            reconf_opts=reconf_opts,
            progbar=False,
        )
    elif kind == "forest-reconfigure":
        tree = tree.slice_and_reconfigure_forest(
            target_size=slicing["target_size"],
            step_size=slicing["step_size"],
            num_trees=slicing["num_trees"],
            max_repeats=slicing["max_repeats"],
            allow_outer=slicing["allow_outer"],
            minimize=minimize,
            parallel=parallel,
            progbar=False,
        )
    else:
        raise ValueError(f"unknown cotengra slicing kind: {kind}")

    sliced = [
        {
            "label": info.ind,
            "inner": info.inner,
            "size": info.size,
            "project": info.project,
        }
        for info in tree.sliced_inds.values()
    ]
    json.dump({"path": tree.get_path(), "sliced": sliced}, sys.stdout)

try:
    main()
except Exception as exc:
    json.dump(
        {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
        sys.stdout,
    )
    sys.exit(1)
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CotengraPythonOptimizer {
    config: CotengraPythonConfig,
}

impl CotengraPythonOptimizer {
    pub fn new(config: CotengraPythonConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &CotengraPythonConfig {
        &self.config
    }

    /// Search a cotengra path and, if configured, cotengra slicing decision,
    /// then package both as a TeNeT [`SlicedPlan`]. This is planner-only: TeNeT's
    /// ordinary tensor executor does not execute the slices yet.
    pub fn optimize_sliced(
        &self,
        ir: &NetworkIR,
        cost_model: &DenseCostModel,
    ) -> Result<SlicedPlan> {
        if ir.tensors().len() < 2 {
            return Err(ContractError::NotEnoughTensors);
        }
        let spec = cotengra_spec(ir, cost_model, &self.config);
        let result = run_cotengra_python(&self.config, &spec)?;
        let pairs = path_to_active_pairs(&result.path)?;
        let plan = ContractionPlan::from_dense_active_pair_path(ir, &pairs, cost_model)?;
        let sliced = parse_sliced_labels(ir, cost_model, &result.sliced)?;
        let slice = slice_plan_for_ordered(ir, &plan, cost_model, &sliced);
        Ok(SlicedPlan::new(plan, slice))
    }
}

impl Default for CotengraPythonOptimizer {
    fn default() -> Self {
        Self::new(CotengraPythonConfig::default())
    }
}

impl DenseContractionOptimizer for CotengraPythonOptimizer {
    fn optimize(
        &self,
        ir: &NetworkIR,
        cost_model: &DenseCostModel,
    ) -> Result<Vec<ContractionStep>> {
        if ir.tensors().len() < 2 {
            return Err(ContractError::NotEnoughTensors);
        }

        let mut config = self.config.clone();
        config.slicing = CotengraSlicingConfig::None;
        let spec = cotengra_spec(ir, cost_model, &config);
        let result = run_cotengra_python(&config, &spec)?;
        let pairs = path_to_active_pairs(&result.path)?;
        dense_steps_from_active_pair_path(ir, &pairs, cost_model)
    }
}

fn cotengra_spec(
    ir: &NetworkIR,
    cost_model: &DenseCostModel,
    config: &CotengraPythonConfig,
) -> Value {
    let inputs = ir
        .tensors()
        .iter()
        .map(|tensor| {
            tensor
                .labels()
                .iter()
                .map(|label| label.as_str())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let output = ir
        .output_labels()
        .iter()
        .map(|label| label.as_str())
        .collect::<Vec<_>>();

    let mut size_dict = BTreeMap::new();
    for tensor in ir.tensors() {
        for label in tensor.labels() {
            size_dict.insert(label.as_str(), cost_model.dim(label).unwrap_or(1));
        }
    }
    for label in ir.output_labels() {
        size_dict.insert(label.as_str(), cost_model.dim(label).unwrap_or(1));
    }

    json!({
        "inputs": inputs,
        "output": output,
        "size_dict": size_dict,
        "config": {
            "method": method_name(&config.method),
            "minimize": minimize_name(&config.minimize),
            "max_repeats": config.max_repeats.max(1),
            "seed": config.seed,
            "parallel": config.parallel,
            "slicing": slicing_spec(&config.slicing),
        },
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CotengraPythonResult {
    path: Vec<Vec<usize>>,
    sliced: Vec<CotengraSlicedIndex>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CotengraSlicedIndex {
    label: String,
    inner: bool,
    project: Option<usize>,
}

fn run_cotengra_python(
    config: &CotengraPythonConfig,
    spec: &Value,
) -> Result<CotengraPythonResult> {
    let command = python_command(config);
    let command_text = command_text(&command);
    let mut child = Command::new(&command.program)
        .args(&command.args)
        .arg("-c")
        .arg(PYTHON_PLANNER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| {
            ContractError::InvalidContractionPlan(format!(
                "failed to start cotengra Python planner `{command_text}`: {err}"
            ))
        })?;

    {
        let stdin = child.stdin.as_mut().ok_or_else(|| {
            ContractError::InvalidContractionPlan(
                "failed to open cotengra Python planner stdin".to_string(),
            )
        })?;
        let bytes = serde_json::to_vec(spec).map_err(|err| {
            ContractError::InvalidContractionPlan(format!(
                "failed to serialize cotengra planner input: {err}"
            ))
        })?;
        stdin.write_all(&bytes).map_err(|err| {
            ContractError::InvalidContractionPlan(format!(
                "failed to write cotengra planner input: {err}"
            ))
        })?;
    }

    let output = child.wait_with_output().map_err(|err| {
        ContractError::InvalidContractionPlan(format!(
            "failed to wait for cotengra Python planner: {err}"
        ))
    })?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let value: Value = serde_json::from_slice(&output.stdout).map_err(|err| {
        ContractError::InvalidContractionPlan(format!(
            "cotengra Python planner returned non-JSON stdout: {err}; stdout={stdout:?}; stderr={stderr:?}"
        ))
    })?;

    if !output.status.success() {
        let message = value
            .get("traceback")
            .or_else(|| value.get("error"))
            .and_then(Value::as_str)
            .unwrap_or(stdout.as_ref());
        return Err(ContractError::InvalidContractionPlan(format!(
            "cotengra Python planner failed: {message}; stderr={stderr:?}"
        )));
    }

    parse_planner_output(&value)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PythonCommand {
    program: String,
    args: Vec<String>,
}

fn python_command(config: &CotengraPythonConfig) -> PythonCommand {
    if let Some(program) = config
        .python
        .clone()
        .or_else(|| std::env::var("TENET_COTENGRA_PYTHON").ok())
    {
        return PythonCommand {
            program,
            args: config.python_args.clone(),
        };
    }

    if let Ok(project) = std::env::var("TENET_COTENGRA_UV_PROJECT") {
        return PythonCommand {
            program: "uv".to_string(),
            args: vec![
                "run".to_string(),
                "--project".to_string(),
                resolve_cotengra_uv_project(project),
                "python".to_string(),
            ],
        };
    }

    PythonCommand {
        program: "python3".to_string(),
        args: Vec::new(),
    }
}

fn command_text(command: &PythonCommand) -> String {
    std::iter::once(command.program.as_str())
        .chain(command.args.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}

fn resolve_cotengra_uv_project(project: String) -> String {
    let path = std::path::Path::new(&project);
    if path.is_absolute() || path.exists() {
        return project;
    }

    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    if let Some(workspace) = manifest.parent() {
        let workspace_path = workspace.join(&project);
        if workspace_path.exists() {
            return workspace_path.to_string_lossy().into_owned();
        }
    }

    project
}

fn parse_planner_output(value: &Value) -> Result<CotengraPythonResult> {
    Ok(CotengraPythonResult {
        path: parse_path(value)?,
        sliced: parse_sliced(value)?,
    })
}

fn parse_path(value: &Value) -> Result<Vec<Vec<usize>>> {
    let path = value.get("path").ok_or_else(|| {
        ContractError::InvalidContractionPlan(
            "cotengra Python planner output is missing `path`".to_string(),
        )
    })?;
    let path = path.as_array().ok_or_else(|| {
        ContractError::InvalidContractionPlan(
            "cotengra Python planner `path` is not an array".to_string(),
        )
    })?;
    path.iter()
        .map(|step| {
            let step = step.as_array().ok_or_else(|| {
                ContractError::InvalidContractionPlan(
                    "cotengra Python planner path step is not an array".to_string(),
                )
            })?;
            step.iter()
                .map(|index| {
                    index.as_u64().map(|value| value as usize).ok_or_else(|| {
                        ContractError::InvalidContractionPlan(
                            "cotengra Python planner path index is not an unsigned integer"
                                .to_string(),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect()
}

fn parse_sliced(value: &Value) -> Result<Vec<CotengraSlicedIndex>> {
    let Some(sliced) = value.get("sliced") else {
        return Ok(Vec::new());
    };
    let sliced = sliced.as_array().ok_or_else(|| {
        ContractError::InvalidContractionPlan(
            "cotengra Python planner `sliced` is not an array".to_string(),
        )
    })?;
    sliced
        .iter()
        .map(|entry| {
            let label = entry
                .get("label")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ContractError::InvalidContractionPlan(
                        "cotengra Python planner sliced entry is missing string `label`"
                            .to_string(),
                    )
                })?
                .to_string();
            let inner = entry.get("inner").and_then(Value::as_bool).ok_or_else(|| {
                ContractError::InvalidContractionPlan(
                    "cotengra Python planner sliced entry is missing bool `inner`".to_string(),
                )
            })?;
            let project = match entry.get("project") {
                None | Some(Value::Null) => None,
                Some(value) => Some(value.as_u64().ok_or_else(|| {
                    ContractError::InvalidContractionPlan(
                        "cotengra Python planner sliced `project` is not an unsigned integer"
                            .to_string(),
                    )
                })? as usize),
            };
            Ok(CotengraSlicedIndex {
                label,
                inner,
                project,
            })
        })
        .collect()
}

fn parse_sliced_labels(
    ir: &NetworkIR,
    cost_model: &DenseCostModel,
    sliced: &[CotengraSlicedIndex],
) -> Result<Vec<TemporaryLabel>> {
    let mut labels = Vec::with_capacity(sliced.len());
    for index in sliced {
        if index.project.is_some() {
            return Err(ContractError::InvalidContractionPlan(format!(
                "cotengra returned projected index `{}`; TeNeT sliced plans only support full slices",
                index.label
            )));
        }
        let label = TemporaryLabel::from(index.label.as_str());
        if cost_model.dim(&label).is_none() {
            return Err(ContractError::InvalidContractionPlan(format!(
                "cotengra returned unknown sliced index `{}`",
                index.label
            )));
        }
        let expected = if ir.output_labels().contains(&label) {
            SliceKind::Output
        } else {
            SliceKind::Internal
        };
        let actual = if index.inner {
            SliceKind::Internal
        } else {
            SliceKind::Output
        };
        if actual != expected {
            return Err(ContractError::InvalidContractionPlan(format!(
                "cotengra sliced index `{}` kind mismatch: cotengra={actual:?} TeNeT={expected:?}",
                index.label
            )));
        }
        labels.push(label);
    }
    Ok(labels)
}

fn path_to_active_pairs(path: &[Vec<usize>]) -> Result<Vec<ActivePair>> {
    path.iter()
        .map(|step| match step.as_slice() {
            [lhs, rhs] => Ok(ActivePair::new(*lhs, *rhs)),
            other => Err(ContractError::InvalidContractionPlan(format!(
                "cotengra returned a non-pairwise step with {} operands; TeNeT plans are strictly pairwise",
                other.len()
            ))),
        })
        .collect()
}

fn method_name(method: &CotengraPythonMethod) -> &'static str {
    match method {
        CotengraPythonMethod::Auto => "auto",
        CotengraPythonMethod::AutoHq => "auto-hq",
        CotengraPythonMethod::Greedy => "greedy",
        CotengraPythonMethod::Optimal => "optimal",
        CotengraPythonMethod::RandomGreedy => "random-greedy",
        CotengraPythonMethod::Hyper => "hyper",
    }
}

fn minimize_name(minimize: &CotengraMinimize) -> &str {
    match minimize {
        CotengraMinimize::Flops => "flops",
        CotengraMinimize::Size => "size",
        CotengraMinimize::Write => "write",
        CotengraMinimize::Combo => "combo",
        CotengraMinimize::Limit => "limit",
        CotengraMinimize::Custom(value) => value.as_str(),
    }
}

fn slicing_spec(slicing: &CotengraSlicingConfig) -> Value {
    match slicing {
        CotengraSlicingConfig::None => json!({"kind": "none"}),
        CotengraSlicingConfig::Slice {
            target_size,
            max_repeats,
            allow_outer,
        } => json!({
            "kind": "slice",
            "target_size": target_size,
            "max_repeats": max_repeats.max(&1),
            "allow_outer": allow_outer,
        }),
        CotengraSlicingConfig::Reconfigure {
            target_size,
            step_size,
            max_repeats,
            allow_outer,
            forested,
        } => json!({
            "kind": "reconfigure",
            "target_size": target_size,
            "step_size": step_size.max(&1),
            "max_repeats": max_repeats.max(&1),
            "allow_outer": allow_outer,
            "forested": forested,
        }),
        CotengraSlicingConfig::ForestReconfigure {
            target_size,
            step_size,
            num_trees,
            max_repeats,
            allow_outer,
        } => json!({
            "kind": "forest-reconfigure",
            "target_size": target_size,
            "step_size": step_size.max(&1),
            "num_trees": num_trees.max(&1),
            "max_repeats": max_repeats.max(&1),
            "allow_outer": allow_outer,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_einsum;
    use crate::DenseTensorInfo;

    #[test]
    fn spec_preserves_labels_and_dims() {
        let ir = parse_einsum("ab,bc->ac").unwrap();
        let infos = vec![
            DenseTensorInfo::new(vec![2, 3]),
            DenseTensorInfo::new(vec![3, 4]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        let spec = cotengra_spec(&ir, &cost, &CotengraPythonConfig::default());
        assert_eq!(spec["inputs"], json!([["a", "b"], ["b", "c"]]));
        assert_eq!(spec["output"], json!(["a", "c"]));
        assert_eq!(spec["size_dict"], json!({"a": 2, "b": 3, "c": 4}));
        assert_eq!(spec["config"]["method"], json!("auto-hq"));
        assert_eq!(spec["config"]["slicing"], json!({"kind": "none"}));
    }

    #[test]
    fn spec_encodes_reconfigure_slicing() {
        let ir = parse_einsum("ab,bc->ac").unwrap();
        let infos = vec![
            DenseTensorInfo::new(vec![2, 3]),
            DenseTensorInfo::new(vec![3, 4]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        let mut config = CotengraPythonConfig::default();
        config.slicing = CotengraSlicingConfig::Reconfigure {
            target_size: 8,
            step_size: 2,
            max_repeats: 7,
            allow_outer: false,
            forested: true,
        };
        let spec = cotengra_spec(&ir, &cost, &config);
        assert_eq!(
            spec["config"]["slicing"],
            json!({
                "kind": "reconfigure",
                "target_size": 8,
                "step_size": 2,
                "max_repeats": 7,
                "allow_outer": false,
                "forested": true,
            })
        );
    }

    #[test]
    fn parses_path_and_sliced_indices() {
        let value = json!({
            "path": [[0, 1], [0, 1]],
            "sliced": [
                {"label": "a", "inner": false, "size": 2, "project": null},
                {"label": "b", "inner": true, "size": 3, "project": null},
            ],
        });
        let parsed = parse_planner_output(&value).unwrap();
        assert_eq!(parsed.path, vec![vec![0, 1], vec![0, 1]]);
        assert_eq!(
            parsed.sliced,
            vec![
                CotengraSlicedIndex {
                    label: "a".to_string(),
                    inner: false,
                    project: None,
                },
                CotengraSlicedIndex {
                    label: "b".to_string(),
                    inner: true,
                    project: None,
                },
            ]
        );
    }

    #[test]
    fn rejects_non_pairwise_path_steps() {
        let err = path_to_active_pairs(&[vec![0], vec![0, 1, 2]]).unwrap_err();
        assert!(err.to_string().contains("non-pairwise"));
    }

    #[test]
    fn uv_project_config_builds_python_command() {
        let config = CotengraPythonConfig::with_uv_project("tools/cotengra-python");
        let expected_project = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("tools/cotengra-python")
            .to_string_lossy()
            .into_owned();
        assert_eq!(
            python_command(&config),
            PythonCommand {
                program: "uv".to_string(),
                args: vec![
                    "run".to_string(),
                    "--project".to_string(),
                    expected_project,
                    "python".to_string(),
                ],
            }
        );
    }

    #[test]
    fn runs_installed_cotengra_when_requested() {
        if std::env::var_os("TENET_RUN_COTENGRA_PYTHON_TEST").is_none() {
            return;
        }

        let ir = parse_einsum("ab,bc->ac").unwrap();
        let infos = vec![
            DenseTensorInfo::new(vec![2, 3]),
            DenseTensorInfo::new(vec![3, 4]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        let config = std::env::var("TENET_COTENGRA_UV_PROJECT")
            .map(CotengraPythonConfig::with_uv_project)
            .unwrap_or_default();

        let steps = CotengraPythonOptimizer::new(config)
            .optimize(&ir, &cost)
            .unwrap();
        assert_eq!(steps.len(), 1);
    }
}
