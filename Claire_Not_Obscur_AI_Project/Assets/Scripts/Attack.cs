using UnityEngine;

public class Attack : MonoBehaviour
{
    public KeyCode attackKey;
    public Health health;
    private Health opponentInRange;
    public NPCBehaviour NPC;
    public Movement opponentMovement;

    private void OnTriggerEnter(Collider collider)
    {
        
        Health opponent = collider.GetComponentInParent<Health>();

        if (opponent != null && opponent.player != this.GetComponentInParent<Health>().player && collider.tag == "HurtBox" )
        {
            Debug.Log("Collider Tag :: Attack.OnTriggerEnter == " + collider.tag);
            Debug.Log("Collider player name :: Attack OnTriggerEnter == " + opponent.player);
            Debug.Log("Collider position :: Attack.OnTriggerEnter == " + collider.transform.position);

            opponentInRange = opponent; // store reference
            opponentInRange.DecrementHP();
            if (NPC != null)
            {
                NPC.state = NPCBehaviour.State.Hit;
            }
            else
            {
                opponentMovement.isHit = true;
            }
        }
    }

    private void OnTriggerExit(Collider collider)
    {
        Health opponent = collider.gameObject.GetComponent<Health>();
        if (opponent == opponentInRange)
        {
            opponentInRange = null; // clear when they leave range
        }
    }
}
