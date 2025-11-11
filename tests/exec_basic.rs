#[cfg(feature = "cpu-exec")]
mod cpu {
    use mind::eval;
    use mind::parser;

    use std::collections::HashMap;

    #[test]
    fn add_scalar_exec() {
        let src = "let x: Tensor[f32,(2,2)] = 1; x + 2";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval::eval_module_value_with_env_mode(
            &module,
            &mut env,
            Some(src),
            eval::ExecMode::CpuExec,
        )
        .unwrap();
        let text = eval::format_value_human(&value);
        assert!(text.contains("(2,2)"));
        assert!(text.contains("materialized"));
    }

    #[test]
    fn matmul_exec() {
        let src = "let a: Tensor[f32,(2,2)] = 1; let b: Tensor[f32,(2,2)] = 1; tensor.matmul(a,b)";
        let module = parser::parse(src).unwrap();
        let mut env = HashMap::new();
        let value = eval::eval_module_value_with_env_mode(
            &module,
            &mut env,
            Some(src),
            eval::ExecMode::CpuExec,
        )
        .unwrap();
        let text = eval::format_value_human(&value);
        assert!(text.contains("(2,2)"));
        assert!(text.contains("materialized"));
    }
}
