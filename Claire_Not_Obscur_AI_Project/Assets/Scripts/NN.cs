using UnityEngine;
using Unity.InferenceEngine;

public class AIController : MonoBehaviour
{
    public ModelAsset modelAsset;
    private Model runtimeModel;
    private Worker worker;

    public GameObject player1;
    public GameObject player2;
    public HPBar player1HPBar;
    public HPBar player2HPBar;

    // Feature names for reference
    private readonly string[] featureNames = new string[]
    {
        "P1HP", "P1XCoord", "P1YCoord",
        "P1IsGrounded", "P1IsDashing", "P1IsAttacking",
        "P1IsHit", "P1IsInHitStun", "P1IsLocked",
        "P2HP", "P2XCoord", "P2YCoord",
        "P2IsGrounded", "P2IsDashing", "P2IsAttacking",
        "P2IsHit", "P2IsInHitStun", "P2IsLocked",
        "Distance"
    };

    // Action names
    private readonly string[] actionNames = new string[]
    {
        "Left", "Right", "Jump", "Dash", "Attack", "Down"
    };

    void Start()
    {
        // Load the model
        if (modelAsset == null)
        {
            Debug.LogError("Model asset not assigned!");
            return;
        }

        runtimeModel = ModelLoader.Load(modelAsset);

        // Create a worker (GPUCompute for better performance, CPU for compatibility)
        worker = new Worker(runtimeModel, BackendType.GPUCompute);

        Debug.Log("Model loaded successfully!");
    }

    void Update() {
        Movement player1Movement = player1.GetComponent<Movement>();
        Movement player2Movement = player2.GetComponent<Movement>();
        float[] state = new float[19] {
            (float)player1HPBar.health.currentHP,
            (float)player1.transform.position.x,
            (float)player1.transform.position.y,
            (float)System.Convert.ToSingle(player1Movement.isGrounded),
            (float)System.Convert.ToSingle(player1Movement.dashing),
            (float)System.Convert.ToSingle(player1Movement.attacking),
            (float)System.Convert.ToSingle(player1Movement.isHit),
            (float)System.Convert.ToSingle(player1Movement.inHitstun),
            (float)System.Convert.ToSingle(player1Movement.locked),
            (float)player2HPBar.health.currentHP,
            (float)player2.transform.position.x,
            (float)player2.transform.position.y,
            (float)System.Convert.ToSingle(player2Movement.isGrounded),
            (float)System.Convert.ToSingle(player2Movement.dashing),
            (float)System.Convert.ToSingle(player2Movement.attacking),
            (float)System.Convert.ToSingle(player2Movement.isHit),
            (float)System.Convert.ToSingle(player2Movement.inHitstun),
            (float)System.Convert.ToSingle(player2Movement.locked),
            (float)(player1.transform.position - player2.transform.position).magnitude
        };

        float[] actions = Predict(state);

        // for (int i = 0; i < featureNames.Length; i++)
        // {
        //     Debug.Log($"{featureNames[i]}: {state[i]}");
        // }

        // for (int i = 0; i < actionNames.Length; i++)
        // {
        //     Debug.Log($"{actionNames[i]}: {actions[i]}");
        //     if (actions[i] > 0.5f) {
        //         Debug.Log($"{actionNames[i]}");
        //     }
        // }

        float sensitivity = 0.12f;
        player2Movement.leftPressed = actions[0] > sensitivity;
        player2Movement.rightPressed = actions[1] > sensitivity;
        if (player2Movement.leftPressed && player2Movement.rightPressed) {
            if (actions[0] > actions[1]) {
                player2Movement.rightPressed = false;
            }
            else {
                player2Movement.leftPressed = false;
            }
        }
        player2Movement.jumpPressed = actions[2] > sensitivity;
        player2Movement.dashPressed = actions[3] > sensitivity;
        player2Movement.attackPressed = actions[4] > sensitivity;
        player2Movement.downPressed = actions[5] > sensitivity;
    }

    public float[] Predict(float[] gameState)
    {
        if (gameState.Length != 19)
        {
            Debug.LogError($"Invalid input size. Expected 19, got {gameState.Length}");
            return new float[6];
        }

        // Normalize the input
        float[] normalized = NormalizeInput(gameState);

        // Create input tensor (batch size = 1, features = 19)
        TensorShape inputShape = new TensorShape(1, 19);
        Tensor<float> inputTensor = new Tensor<float>(inputShape, normalized);

        // Run inference
        worker.Schedule(inputTensor);

        // Get output
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;

        // Copy results to array
        // float[] results = new float[6];
        // outputTensor.ReadBackAndClone();
        // for (int i = 0; i < 6; i++)
        // {
        //     results[i] = outputTensor[i];
        // }
        float[] results = outputTensor.DownloadToArray();

        // Dispose input tensor
        inputTensor.Dispose();

        return results;
    }

    private float[] NormalizeInput(float[] input)
    {
        float[] normalized = new float[19];

        // P1HP (index 0): divide by 4
        normalized[0] = input[0] / 4f;

        // P1XCoord (index 1): divide by 8, clamp to [-1, 1]
        normalized[1] = Mathf.Clamp(input[1] / 8f, -1f, 1f);

        // P1YCoord (index 2): divide by 5, clamp to [-1, 1]
        normalized[2] = Mathf.Clamp(input[2] / 5f, -1f, 1f);

        // P1 boolean flags (indices 3-8): no normalization
        for (int i = 3; i <= 8; i++)
        {
            normalized[i] = input[i];
        }

        // P2HP (index 9): divide by 4
        normalized[9] = input[9] / 4f;

        // P2XCoord (index 10): divide by 8, clamp to [-1, 1]
        normalized[10] = Mathf.Clamp(input[10] / 8f, -1f, 1f);

        // P2YCoord (index 11): divide by 5, clamp to [-1, 1]
        normalized[11] = Mathf.Clamp(input[11] / 5f, -1f, 1f);

        // P2 boolean flags (indices 12-17): no normalization
        for (int i = 12; i <= 17; i++)
        {
            normalized[i] = input[i];
        }

        // Distance (index 18): divide by 16, clamp to [-1, 1]
        normalized[18] = Mathf.Clamp(input[18] / 16f, -1f, 1f);

        return normalized;
    }

    void OnDestroy()
    {
        // Clean up resources
        worker?.Dispose();
    }

    // Example: Use this in your game loop to get AI actions
    public bool[] GetAIActions(float[] gameState)
    {
        float[] predictions = Predict(gameState);
        bool[] actions = new bool[6];

        // Threshold at 0.5
        for (int i = 0; i < 6; i++)
        {
            actions[i] = predictions[i] > 0.5f;
        }

        return actions;
    }
}
