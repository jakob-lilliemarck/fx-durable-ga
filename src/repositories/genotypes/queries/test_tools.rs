use crate::repositories::genotypes::Genotype;
use crate::repositories::genotypes::store_genotypes;
use crate::services::evaluation::repositories::evaluations::Evaluation;
use crate::services::evaluation::repositories::evaluations::queries::store_evaluations;
use crate::services::optimization::store_request;
use crate::services::optimization::{FitnessGoal, Request, Schedule, Selector};
use chrono::Utc;
use uuid::Uuid;

pub(super) async fn seed(pool: &sqlx::PgPool) -> (Uuid, Uuid, [Uuid; 5]) {
    // Create requests first
    let request_1 = Request::new(
        "test",
        FitnessGoal::maximize(0.9).unwrap(),
        Selector::tournament(10),
        Schedule::generational(100, 10),
    );
    let request_2 = Request::new(
        "test",
        FitnessGoal::maximize(0.9).unwrap(),
        Selector::tournament(10),
        Schedule::generational(100, 10),
    );

    let rid_1 = request_1.id;
    let rid_2 = request_2.id;

    store_request(pool, request_1).await.unwrap();
    store_request(pool, request_2).await.unwrap();

    let genotypes = vec![
        Genotype::new(
            "test",
            serde_json::json!([1, 2, 3]),
            rid_1,
            Some(1),
            None,
            None,
        )
        .unwrap(),
        Genotype::new(
            "test",
            serde_json::json!([4, 5, 6]),
            rid_1,
            Some(2),
            None,
            None,
        )
        .unwrap(),
        Genotype::new(
            "test",
            serde_json::json!([7, 8, 9]),
            rid_2,
            Some(1),
            None,
            None,
        )
        .unwrap(),
        Genotype::new(
            "test",
            serde_json::json!([10, 11, 12]),
            rid_2,
            Some(1),
            None,
            None,
        )
        .unwrap(),
        Genotype::new(
            "test",
            serde_json::json!([13, 14, 15]),
            rid_2,
            Some(2),
            None,
            None,
        )
        .unwrap(),
    ];

    let gid_1 = genotypes[0].id;
    let gid_2 = genotypes[1].id;
    let gid_3 = genotypes[2].id;
    let gid_4 = genotypes[3].id;
    let gid_5 = genotypes[4].id;

    store_genotypes(pool, &genotypes).await.unwrap();

    let host_id = Uuid::now_v7();

    let evaluations = vec![
        Evaluation::new(
            gid_1,
            Uuid::nil(),
            "test".to_string(),
            0.11,
            Some(Utc::now()),
            Some(Utc::now()),
            Some(host_id.clone()),
        ),
        Evaluation::new(
            gid_3,
            Uuid::nil(),
            "test".to_string(),
            0.12,
            Some(Utc::now()),
            Some(Utc::now()),
            Some(host_id.clone()),
        ),
        Evaluation::new(
            gid_4,
            Uuid::nil(),
            "test".to_string(),
            0.42,
            Some(Utc::now()),
            Some(Utc::now()),
            Some(host_id.clone()),
        ),
    ];

    store_evaluations(pool, &evaluations).await.unwrap();
    // genotype_id_2 and genotype_id_5 have no fitness
    (rid_1, rid_2, [gid_1, gid_2, gid_3, gid_4, gid_5])
}

pub(super) async fn seed_lineage(pool: &sqlx::PgPool) -> Vec<Uuid> {
    let request = Request::new(
        "lineage",
        FitnessGoal::maximize(0.9).unwrap(),
        Selector::tournament(10),
        Schedule::generational(100, 10),
    );
    let request_id = request.id;
    store_request(pool, request).await.unwrap();

    let root = Genotype::new(
        "lineage",
        serde_json::json!([0]),
        request_id,
        Some(1),
        None,
        None,
    )
    .unwrap();
    let root_id = root.id();

    let child = Genotype::new(
        "lineage",
        serde_json::json!([1]),
        request_id,
        Some(2),
        Some(&root_id),
        None,
    )
    .unwrap();
    let child_id = child.id();

    let grandchild = Genotype::new(
        "lineage",
        serde_json::json!([2]),
        request_id,
        Some(3),
        Some(&child_id),
        None,
    )
    .unwrap();
    let grandchild_id = grandchild.id();

    store_genotypes(pool, &[root, child, grandchild])
        .await
        .unwrap();

    let host_id = Uuid::now_v7();
    for genotype_id in [root_id, child_id, grandchild_id] {
        store_evaluations(
            pool,
            &[Evaluation::new(
                genotype_id,
                Uuid::nil(),
                "test".to_string(),
                0.5,
                Some(Utc::now()),
                Some(Utc::now()),
                Some(host_id),
            )],
        )
        .await
        .unwrap();
    }

    // indexes: [0] root, [1] child, [2] grandchild
    vec![root_id, child_id, grandchild_id]
}
