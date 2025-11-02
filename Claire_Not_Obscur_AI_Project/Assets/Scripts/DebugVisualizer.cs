using UnityEngine;

public class DebugVisualizer : MonoBehaviour
{
    [SerializeField] BoxCollider box;
     
    private void Start()
    {
        
    }
    private void OnDrawGizmos()
    {
        Gizmos.color = Color.red;
        Gizmos.matrix = Matrix4x4.TRS(transform.position, transform.rotation, transform.localScale);
       
        Gizmos.DrawWireCube(box.center, box.size);
    }
}
