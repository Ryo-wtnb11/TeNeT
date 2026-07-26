//! The frozen source/projection contract that every shipped table carries.
//!
//! See `references.md` for the full provenance tables and their verification
//! record; the types here are the machine-readable form of the same contract.

/// Identifier of the one exact-source -> [`num_complex::Complex64`] conversion
/// convention this crate implements.
///
/// The convention itself is spelled out in `references.md`; in one line it is
/// "split on whitespace, `str::parse::<f64>()` each of the trailing two
/// decimal literals, pair as `re + i·im`, absent records are exact zero".
///
/// Bump this string whenever that conversion changes in any observable way.
/// It is folded into every provider's [`tenet_sectors::RuleIdentity`], so a
/// bump makes previously-built identities compare unequal — which is the
/// point: cached recoupling data computed under the old projection must not be
/// reused under the new one.
pub const PROJECTION_EPOCH: &str = "tenet-category-data/projection/2026-07-27";

/// Immutable lineage of one shipped category table.
///
/// Every field is part of the category's identity. The struct is deliberately
/// a flat record of `&'static str`: the values are frozen at authoring time,
/// there is nothing to configure, and the canonical serialisation
/// ([`CategoryProvenance::write_canonical_bytes`]) must stay trivially
/// auditable because it feeds `RuleIdentity`.
///
/// Why not a builder, a `Default`, or public construction: a provenance record
/// that a caller can assemble is a provenance record that can lie. Values are
/// authored next to the data they describe and nowhere else.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct CategoryProvenance {
    /// Repository holding the exact algebraic expressions (layer 1).
    pub exact_source_repository: &'static str,
    /// Commit pinned in that repository.
    pub exact_source_commit: &'static str,
    /// How that commit's export program turned exact expressions into decimals.
    pub exact_source_export: &'static str,
    /// Repository that redistributes the exported decimals (layer 3).
    pub distribution_repository: &'static str,
    /// Commit pinned in the distribution repository.
    pub distribution_commit: &'static str,
    /// Distribution package version at that commit.
    pub distribution_version: &'static str,
    /// Julia artifact tag the tables were taken from.
    pub artifact_tag: &'static str,
    /// `git-tree-sha1` recorded for that artifact.
    pub artifact_git_tree_sha1: &'static str,
    /// SHA-256 of the artifact tarball, verified on download.
    pub artifact_tarball_sha256: &'static str,
    /// URL the verified tarball came from.
    pub artifact_url: &'static str,
    /// Upstream category identifier, e.g. `Fib = PMFC{2,1,0,2,0,0}`.
    pub category_key: &'static str,
    /// Frozen object ordering, stated in upstream's one-based labels.
    pub object_order: &'static str,
    /// What was and was not done to the source gauge.
    pub gauge: &'static str,
    /// The projection epoch in force, normally [`PROJECTION_EPOCH`].
    pub projection_epoch: &'static str,
}

impl CategoryProvenance {
    /// Appends the canonical, order-fixed byte serialisation of this record.
    ///
    /// The format is one `key=value` line per field in declaration order. It
    /// is a byte contract, not a display form: it is hashed and compared, so
    /// reordering fields, renaming a key, or changing a separator changes
    /// every derived [`tenet_sectors::RuleIdentity`].
    ///
    /// Why not `Debug` or a serde encoding: `Debug` output is explicitly not a
    /// stability guarantee, and a serialisation framework would put the byte
    /// layout of an identity key behind a dependency's version policy.
    pub fn write_canonical_bytes(&self, out: &mut Vec<u8>) {
        let Self {
            exact_source_repository,
            exact_source_commit,
            exact_source_export,
            distribution_repository,
            distribution_commit,
            distribution_version,
            artifact_tag,
            artifact_git_tree_sha1,
            artifact_tarball_sha256,
            artifact_url,
            category_key,
            object_order,
            gauge,
            projection_epoch,
        } = self;
        // Destructured above so that adding a field without extending this
        // list fails to compile: a silently unhashed provenance field would
        // let two different categories share one identity.
        for (key, value) in [
            ("exact_source_repository", exact_source_repository),
            ("exact_source_commit", exact_source_commit),
            ("exact_source_export", exact_source_export),
            ("distribution_repository", distribution_repository),
            ("distribution_commit", distribution_commit),
            ("distribution_version", distribution_version),
            ("artifact_tag", artifact_tag),
            ("artifact_git_tree_sha1", artifact_git_tree_sha1),
            ("artifact_tarball_sha256", artifact_tarball_sha256),
            ("artifact_url", artifact_url),
            ("category_key", category_key),
            ("object_order", object_order),
            ("gauge", gauge),
            ("projection_epoch", projection_epoch),
        ] {
            out.extend_from_slice(key.as_bytes());
            out.push(b'=');
            out.extend_from_slice(value.as_bytes());
            out.push(b'\n');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: CategoryProvenance = CategoryProvenance {
        exact_source_repository: "repo",
        exact_source_commit: "commit",
        exact_source_export: "export",
        distribution_repository: "dist",
        distribution_commit: "dist-commit",
        distribution_version: "0.0.0",
        artifact_tag: "tag",
        artifact_git_tree_sha1: "tree",
        artifact_tarball_sha256: "tar",
        artifact_url: "url",
        category_key: "key",
        object_order: "order",
        gauge: "gauge",
        projection_epoch: PROJECTION_EPOCH,
    };

    #[test]
    fn canonical_bytes_cover_every_field_in_declaration_order() {
        let mut bytes = Vec::new();
        SAMPLE.write_canonical_bytes(&mut bytes);
        let text = String::from_utf8(bytes).expect("ascii");
        let keys: Vec<&str> = text
            .lines()
            .map(|line| line.split('=').next().unwrap())
            .collect();
        assert_eq!(
            keys,
            [
                "exact_source_repository",
                "exact_source_commit",
                "exact_source_export",
                "distribution_repository",
                "distribution_commit",
                "distribution_version",
                "artifact_tag",
                "artifact_git_tree_sha1",
                "artifact_tarball_sha256",
                "artifact_url",
                "category_key",
                "object_order",
                "gauge",
                "projection_epoch",
            ]
        );
        assert!(text.contains(&format!("projection_epoch={PROJECTION_EPOCH}\n")));
    }

    #[test]
    fn every_provenance_field_change_changes_the_bytes() {
        let baseline = {
            let mut bytes = Vec::new();
            SAMPLE.write_canonical_bytes(&mut bytes);
            bytes
        };
        // Mutation check: flipping each field in turn must move the bytes, so
        // no field is decorative.
        let mutants = [
            CategoryProvenance {
                exact_source_repository: "x",
                ..SAMPLE
            },
            CategoryProvenance {
                exact_source_commit: "x",
                ..SAMPLE
            },
            CategoryProvenance {
                exact_source_export: "x",
                ..SAMPLE
            },
            CategoryProvenance {
                distribution_repository: "x",
                ..SAMPLE
            },
            CategoryProvenance {
                distribution_commit: "x",
                ..SAMPLE
            },
            CategoryProvenance {
                distribution_version: "x",
                ..SAMPLE
            },
            CategoryProvenance {
                artifact_tag: "x",
                ..SAMPLE
            },
            CategoryProvenance {
                artifact_git_tree_sha1: "x",
                ..SAMPLE
            },
            CategoryProvenance {
                artifact_tarball_sha256: "x",
                ..SAMPLE
            },
            CategoryProvenance {
                artifact_url: "x",
                ..SAMPLE
            },
            CategoryProvenance {
                category_key: "x",
                ..SAMPLE
            },
            CategoryProvenance {
                object_order: "x",
                ..SAMPLE
            },
            CategoryProvenance {
                gauge: "x",
                ..SAMPLE
            },
            CategoryProvenance {
                projection_epoch: "x",
                ..SAMPLE
            },
        ];
        for mutant in mutants {
            let mut bytes = Vec::new();
            mutant.write_canonical_bytes(&mut bytes);
            assert_ne!(bytes, baseline);
        }
    }

}
