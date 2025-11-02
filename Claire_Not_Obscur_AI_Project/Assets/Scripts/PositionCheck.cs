
using UnityEngine;

public class PositionCheck : MonoBehaviour
{
    [SerializeField] GameObject player1;
    [SerializeField] GameObject player2;


    // Update is called once per frame
    void Update()
    {
        if ((player2.GetComponent<Movement>() != null))
        {
            if (player1.transform.position.x > player2.transform.position.x)
            {
                player1.transform.rotation = Quaternion.Euler(0, 180, 0);
                player2.transform.rotation = Quaternion.Euler(0, 0, 0);
            }
            else
            {
                player1.transform.rotation = Quaternion.Euler(0, 0, 0);
                player2.transform.rotation = Quaternion.Euler(0, 180, 0);
            }
        }
        else if (player2.GetComponent<NPCBehaviour>() != null) {
            if (player1.transform.position.x > player2.transform.position.x)
            {
                player1.transform.rotation = Quaternion.Euler(0, 180, 0);
                player2.transform.rotation = Quaternion.Euler(0, 0, 0);
            }
            else 
            {
                player1.transform.rotation = Quaternion.Euler(0, 0, 0);
                player2.transform.rotation = Quaternion.Euler(0, 180, 0);
            }
        }

        
        
    }
}
