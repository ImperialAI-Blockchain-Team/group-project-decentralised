import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0x0e2d5Ee4A971ca7C5b019624d429A74E8ECd7a9D';

export default new web3.eth.Contract(modelDatabase, address);
