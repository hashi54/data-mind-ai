import sys, io
sys.path.insert(0, 'C:/Users/USER/.gemini/antigravity-ide/scratch/data-mind-ai')

errors = []
print('=== Functional Audit Tests ===')

# 1. Dataset Manager
try:
    from app.data.dataset_manager import dataset_manager
    csv = b'date,revenue,product,region\n2024-01-01,5000,Widget,North\n2024-01-02,6000,Gadget,South\n2024-01-03,4500,Widget,East\n2024-02-01,5500,Gadget,North\n2024-02-15,7000,Widget,South\n'
    tables = dataset_manager.ingest_and_register_file(csv, 'audit_test.csv')
    assert tables[0]['total_rows'] == 5
    df = dataset_manager.get_dataset_df('audit_test')
    assert len(df) == 5
    print('  OK: DatasetManager ingest + retrieve')
except Exception as e:
    print('  ERROR: DatasetManager ->', e)
    errors.append(e)

# 2. DynamicAnalyticsEngine
try:
    import pandas as pd
    from app.analytics.dynamic_analytics import DynamicAnalyticsEngine
    df = pd.DataFrame({'date': ['2024-01-01','2024-01-02','2024-01-03','2024-02-01','2024-02-15'],
                       'revenue': [5000,6000,4500,5500,7000],
                       'product': ['Widget','Gadget','Widget','Gadget','Widget'],
                       'region': ['North','South','East','North','South']})
    roles = DynamicAnalyticsEngine.infer_column_roles(df)
    assert roles['date_col'] == 'date'
    assert roles['primary_metric'] == 'revenue'
    print('  OK: infer_column_roles')
    kpis = DynamicAnalyticsEngine.get_dataset_kpis(df)
    assert kpis['metric_sum'] == 28000.0
    print('  OK: get_dataset_kpis')
    fc = DynamicAnalyticsEngine.run_dynamic_forecast(df, horizon_days=7)
    assert 'forecast_data' in fc and len(fc['forecast_data']) == 7
    print('  OK: run_dynamic_forecast')
    anom = DynamicAnalyticsEngine.run_dynamic_anomalies(df)
    assert 'total_anomalies_detected' in anom
    print('  OK: run_dynamic_anomalies')
except Exception as e:
    print('  ERROR: DynamicAnalytics ->', e)
    errors.append(e)

# 3. Intent Router
try:
    from app.llm.router import IntentRouter
    intent, reason = IntentRouter.route_intent('Why did sales decrease in August?')
    valid_intents = ['SQL_QUERY', 'ML_PREDICTION', 'EDA_ANALYSIS', 'HYBRID_SQL_RAG', 'RAG_DOCS', 'GENERAL_CHAT']
    assert intent in valid_intents
    print('  OK: IntentRouter ->', intent)
except Exception as e:
    print('  ERROR: IntentRouter ->', e)
    errors.append(e)

# 4. AI Analyst Orchestrator
try:
    from app.llm.analyst_agent import BusinessAnalystOrchestrator
    resp = BusinessAnalystOrchestrator.answer_question('What is the data health score?')
    assert 'answer' in resp
    print('  OK: BusinessAnalystOrchestrator EDA question')
    resp2 = BusinessAnalystOrchestrator.answer_question('Hello what can you do?')
    assert 'answer' in resp2
    print('  OK: BusinessAnalystOrchestrator CHAT question')
except Exception as e:
    print('  ERROR: BusinessAnalystOrchestrator ->', e)
    errors.append(e)

# 5. RAG Pipeline
try:
    from app.rag.pipeline import RAGPipeline
    result = RAGPipeline.query('refund policy returns', top_k=2)
    assert 'retrieved_chunks' in result
    chunk_count = len(result['retrieved_chunks'])
    print('  OK: RAGPipeline.query ->', chunk_count, 'chunks')
except Exception as e:
    print('  ERROR: RAGPipeline ->', e)
    errors.append(e)

# 6. ML Models
try:
    from app.ml.registry import registry
    models = registry.list_all_models()
    print('  OK: ModelRegistry ->', len(models), 'models registered')
    from app.ml.inference.churn_predictor import ChurnInferenceEngine
    churn = ChurnInferenceEngine.predict(customer_id=1)
    assert 'churn_probability' in churn
    print('  OK: ChurnPredictor prob=', round(churn['churn_probability'], 3), 'risk=', churn['risk_level'])
    from app.ml.inference.forecaster import SalesForecasterEngine
    fc2 = SalesForecasterEngine.forecast(horizon='30d')
    assert 'total_predicted_revenue' in fc2
    print('  OK: SalesForecaster total=', round(fc2['total_predicted_revenue'], 0))
except Exception as e:
    print('  ERROR: MLModels ->', e)
    errors.append(e)

# 7. Database dynamic table
try:
    from app.database.connection import execute_query
    df_db = execute_query('SELECT COUNT(*) as cnt FROM audit_test;')
    assert df_db.iloc[0]['cnt'] == 5
    print('  OK: execute_query on dynamic table')
except Exception as e:
    print('  ERROR: execute_query ->', e)
    errors.append(e)

# 8. Security validator
try:
    from app.core.security import validate_read_only_sql
    ok, _ = validate_read_only_sql('SELECT * FROM orders LIMIT 10;')
    assert ok == True
    ok2, msg2 = validate_read_only_sql('DROP TABLE orders;')
    assert ok2 == False
    print('  OK: SQL Security (SELECT allowed, DROP blocked)')
except Exception as e:
    print('  ERROR: Security ->', e)
    errors.append(e)

print()
print('=== RESULT:', 8 - len(errors), '/8 test groups PASSED ===')
if not errors:
    print('ALL FUNCTIONAL TESTS PASSED - PROJECT IS HEALTHY!')

# Clean up temporary test dataset
try:
    dataset_manager.delete_dataset('audit_test')
except Exception:
    pass
